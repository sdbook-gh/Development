using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Windows.Forms;

namespace WordEmbedDemo
{
    /// <summary>
    /// 以“独立进程 + 窗口挂靠”方式嵌入本机 Word，并为“只操作我们自己
    /// 新创建的 Word 进程”提供一套安全机制：
    ///
    ///  1) 启动：Process.Start("WINWORD.EXE", "/x") —— /x 强制启动一个
    ///     全新的独立 Word 实例，绝不切到用户已有的 Word 实例。
    ///  2) 锁定：Process.Start 返回的 Process 对象就是“我们所创建的进程”，
    ///     记录它的 Id（PID）作为唯一操作对象 _pid；所有后续操作（挂靠、
    ///     铺满、剥壳、粘贴、退出）都只认这个 PID。
    ///  3) 防御：启动前快照已有 WINWORD 的 PID 集合，启动后校验 p.Id 不在
    ///     该集合内（若被复用则放弃，避免误伤他人进程）。
    ///  4) 窗口：通过 EnumWindows 按 PID + 类名(OpusApp) + 可见性精确定位
    ///     真正的主框架窗口，避免拿到启动画面等过渡窗口；并以内部文档
    ///     视图 _WwG 出现作为“Word 加载完成”的就绪信号。
    ///  5) 退出：Process.GetProcessById(_pid).Kill() —— 只结束我们自己的
    ///     进程，不影响用户其它 Word。
    ///
    /// 窗口挂靠只用 Win32 API + Process，不写 OLE 就地嵌入接口（避免 vtable
    /// 崩溃 0xC0000005）。菜单粘贴通过 oleacc 按 HWND 晚期绑定该进程的
    /// Word.Selection，不使用全局 Word.Application。
    /// </summary>
    public class WordProcessHost : Control
    {
        private Process _wordProcess;   // 仅指向我们自己启动的进程
        private int _pid;               // 锁定的进程 PID（0 = 未启动）
        private IntPtr _hwnd;           // 该进程的主窗口句柄
        private bool _embedded;
        private readonly HashSet<int> _preExisting = new HashSet<int>();

        public WordProcessHost()
        {
            Dock = DockStyle.Fill;
            BackColor = Color.White;
        }

        // ==================== 公共操作 ====================

        /// <summary>启动一个全新的 Word 进程，并挂靠为本面板子窗口。</summary>
        public bool Start()
        {
            try
            {
                Log("==== 新会话开始 ====");
                SnapshotPreExisting();
                Log("已有 WINWORD 进程快照: " + (_preExisting.Count == 0 ? "(无)" : string.Join(",", _preExisting)));

                string winword = FindWinWordExe();
                Log("WINWORD 路径: " + (winword ?? "(未找到)"));
                if (winword == null) return Fail("未找到 WINWORD.EXE，请确认已安装 Microsoft Word。");

                // 生成一个最小空白文档，让 Word 打开真实文档窗口（而非“开始页”）
                string blank = CreateBlankDocx();
                Log("空白文档路径: " + (blank ?? "(创建失败)"));
                if (blank == null) return Fail("无法创建临时空白文档。");

                // /x 强制新实例 + 文档路径：在新实例中打开真实空白文档
                var psi = new ProcessStartInfo(winword, "/x \"" + blank + "\"")
                {
                    UseShellExecute = false,
                    CreateNoWindow = true
                };
                Process p = Process.Start(psi);
                if (p == null) return Fail("启动 Word 失败。");
                Log("Process.Start 返回 PID=" + p.Id);

                // 防御校验：确保返回的进程确实是“新出现”的，而非被复用的已有实例
                if (_preExisting.Contains(p.Id))
                {
                    try { p.Kill(); } catch { }
                    return Fail("Word 复用了已有实例，已放弃，请重试。");
                }

                _wordProcess = p;
                _pid = p.Id;
                Log("锁定新进程 PID=" + _pid);

                return WaitForMainWindow();
            }
            catch (Exception ex)
            {
                return Fail("启动失败：" + ex.Message);
            }
        }

        /// <summary>关闭当前进程并重新启动一个全新的 Word（新建文档）。</summary>
        public void NewDocument()
        {
            Stop();
            Start();
        }

        /// <summary>
        /// 菜单粘贴：按已锁定 HWND 取该 Word 进程的对象模型，调用 Selection.Paste()。
        /// 与隔离前相同的 COM 粘贴路径，但不走全局 Word.Application。
        /// </summary>
        public void Paste()
        {
            if (!IsValid()) return;

            try
            {
                object om = BindWordNativeOm();
                if (om == null)
                {
                    Log("Paste FAIL: AccessibleObjectFromWindow 未返回对象");
                    MessageBox.Show("无法绑定到嵌入的 Word 窗口，粘贴失败。", "提示",
                        MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }

                if (TryPasteSelection(om))
                    return;

                Log("Paste FAIL: 已绑定对象模型，但 Selection.Paste 均失败");
                MessageBox.Show("粘贴失败：Word 未接受 Selection.Paste。", "提示",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
            catch (Exception ex)
            {
                Log("Paste FAIL: " + ex.GetType().Name + ": " + ex.Message);
                MessageBox.Show("粘贴失败：" + ex.Message, "提示",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        /// <summary>对 _WwG（优先）和 OpusApp 调用 AccessibleObjectFromWindow。</summary>
        private object BindWordNativeOm()
        {
            IntPtr doc = FindDocView();
            IntPtr[] candidates = doc != IntPtr.Zero
                ? new[] { doc, _hwnd }
                : new[] { _hwnd };

            for (int i = 0; i < candidates.Length; i++)
            {
                IntPtr h = candidates[i];
                Guid iid = NativeMethods.IID_IDispatch;
                object obj;
                int hr = NativeMethods.AccessibleObjectFromWindow(
                    h, NativeMethods.OBJID_NATIVEOM, ref iid, out obj);
                Log("Paste: AccessibleObjectFromWindow hwnd=0x" + h.ToString("X") +
                    " class=" + NativeMethods.GetWindowClassName(h) +
                    " hr=0x" + hr.ToString("X8") + " obj=" + (obj != null));
                if (hr == 0 && obj != null)
                    return obj;
            }
            return null;
        }

        /// <summary>兼容 Window / Application 两种 NativeOM 入口。</summary>
        private bool TryPasteSelection(object om)
        {
            try
            {
                ComCall(ComGet(om, "Selection"), "Paste");
                Log("Paste: Selection.Paste OK");
                return true;
            }
            catch (Exception ex)
            {
                Log("Paste: Selection.Paste: " + ex.Message);
            }

            try
            {
                object app = ComGet(om, "Application");
                ComCall(ComGet(app, "Selection"), "Paste");
                Log("Paste: Application.Selection.Paste OK");
                return true;
            }
            catch (Exception ex)
            {
                Log("Paste: Application.Selection.Paste: " + ex.Message);
            }

            try
            {
                object win = ComGet(om, "ActiveWindow");
                ComCall(ComGet(win, "Selection"), "Paste");
                Log("Paste: ActiveWindow.Selection.Paste OK");
                return true;
            }
            catch (Exception ex)
            {
                Log("Paste: ActiveWindow.Selection.Paste: " + ex.Message);
            }

            return false;
        }

        private static object ComGet(object target, string name)
        {
            return target.GetType().InvokeMember(name,
                BindingFlags.GetProperty | BindingFlags.Instance, null, target, null);
        }

        private static void ComCall(object target, string name)
        {
            target.GetType().InvokeMember(name,
                BindingFlags.InvokeMethod | BindingFlags.Instance, null, target, null);
        }

        /// <summary>在 OpusApp 子孙窗口中查找可见的文档视图 _WwG。</summary>
        private IntPtr FindDocView()
        {
            IntPtr found = IntPtr.Zero;
            IntPtr visible = IntPtr.Zero;
            NativeMethods.EnumChildWindows(_hwnd, (h, l) =>
            {
                if (NativeMethods.GetWindowClassName(h) != "_WwG") return true;
                if (found == IntPtr.Zero) found = h;
                if (NativeMethods.IsWindowVisible(h))
                {
                    visible = h;
                    return false;
                }
                return true;
            }, IntPtr.Zero);
            return visible != IntPtr.Zero ? visible : found;
        }

        /// <summary>只结束我们自己的 Word 进程（不影响用户其它 Word）。</summary>
        public void Stop()
        {
            if (_pid != 0)
            {
                try
                {
                    var proc = Process.GetProcessById(_pid);
                    if (proc != null && !proc.HasExited)
                        proc.Kill();
                }
                catch (Exception) { }
                _pid = 0;
            }
            _wordProcess = null;
            _hwnd = IntPtr.Zero;
            _embedded = false;
        }

        // ==================== 内部实现 ====================

        /// <summary>启动前快照当前所有 WINWORD 进程的 PID。</summary>
        private void SnapshotPreExisting()
        {
            _preExisting.Clear();
            try
            {
                foreach (var pr in Process.GetProcessesByName("WINWORD"))
                    _preExisting.Add(pr.Id);
            }
            catch { }
        }

        /// <summary>定位 WINWORD.EXE 的完整路径。</summary>
        private static string FindWinWordExe()
        {
            string[] candidates =
            {
                @"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
                @"C:\Program Files (x86)\Microsoft Office\root\Office16\WINWORD.EXE",
                @"C:\Program Files\Microsoft Office\Office16\WINWORD.EXE",
                @"C:\Program Files (x86)\Microsoft Office\Office16\WINWORD.EXE",
            };
            foreach (var c in candidates)
                if (File.Exists(c)) return c;

            // 兜底：从注册表/标准安装路径探测
            try
            {
                string office = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86);
                string p = Path.Combine(office, @"Microsoft Office\root\Office16\WINWORD.EXE");
                if (File.Exists(p)) return p;
            }
            catch { }
            return null;
        }

        /// <summary>在临时目录生成一个最小有效空白 .docx。</summary>
        private static string CreateBlankDocx()
        {
            try
            {
                string dir = Path.Combine(Path.GetTempPath(), "WordEmbedDemo");
                Directory.CreateDirectory(dir);
                string path = Path.Combine(dir, "blank.docx");
                byte[] bytes = Convert.FromBase64String(
                    "UEsDBBQAAAAIAAAAIQAXmADX6wAAALIBAAATAAAAW0NvbnRlbnRfVHlwZXNdLnhtbH1QyU4DMQy98xWRr2gmAweEUKc9sByBQ/kAK/HMRM2mOC3t3+NpoQdUONpvs99itQ9e7aiwS7GHm7YDRdEk6+LYw8f6pbkHxRWjRZ8i9XAghtXyarE+ZGIl4sg9TLXmB63ZTBSQ25QpCjKkErDKWEad0WxwJH3bdXfapFgp1qbOHiBmTzTg1lf1vJf96ZJCnkE9nphzWA+Ys3cGq+B6F+2vmOY7ohXlkcOTy3wtBNCXI2bo74Qf4ZuUU5wl9Y6lvmIQmv5MxWqbzDaItP3f58KlaRicobN+dsslGWKW1oNvz0hAF88f6GPlyy9QSwMEFAAAAAgAAAAhAD+t/vqvAAAALAEAAAsAAABfcmVscy8ucmVsc43POw7CMAwA0J1TRN5pWgaEUEMXhNQVlQNEiZtWNB/F4dPbk4EBKgZG/57tunnaid0x0uidgKoogaFTXo/OCLh0p/UOGCXptJy8QwEzEjSHVX3GSaY8Q8MYiGXEkYAhpbDnnNSAVlLhA7pc6X20MuUwGh6kukqDfFOWWx4/DVigrNUCYqsrYN0c8B/c9/2o8OjVzaJLP3YsOrIso8Ek4OGj5vqdLjILPJ/Dv548vABQSwMEFAAAAAgAAAAhAIWE7cmZAAAAywAAABEAAAB3b3JkL2RvY3VtZW50LnhtbEWOwQ7CIBBE734F2bulejCmKfTmF+gHIGBtUnYJi9b+vVAPXt5kM5OZ7YdPmMXbJ54IFRyaFoRHS27CUcHtetmfQXA26MxM6BWsnmHQu37pHNlX8JhFaUDuFgXPnGMnJdunD4Ybih6L96AUTC5nGuVCycVE1jOXgTDLY9ueZDATgi6Vd3Jr1ViRKrIWvaxSmTbGjb+o/L+hv1BLAwQUAAAACAAAACEAKEYlsH4AAACcAAAAHAAAAHdvcmQvX3JlbHMvZG9jdW1lbnQueG1sLnJlbHNVzEEOwiAQheG9pyCzt6ALYwy0ux7A6AEmdARiOxCGGL29LHX58ud9dnpvq3pRlZTZwWEwoIh9XhIHB/fbvD+Dkoa84JqZHHxIYBp39kortv6RmIqojrA4iK2Vi9biI20oQy7EvTxy3bD1WYMu6J8YSB+NOen6a8Bo9R86fgFQSwECFAMUAAAACAAAACEAF5gA1+sAAACyAQAAEwAAAAAAAAAAAAAAgAEAAAAAW0NvbnRlbnRfVHlwZXNdLnhtbFBLAQIUAxQAAAAIAAAAIQA/rf76rwAAACwBAAALAAAAAAAAAAAAAACAARwBAABfcmVscy8ucmVsc1BLAQIUAxQAAAAIAAAAIQCFhO3JmQAAAMsAAAARAAAAAAAAAAAAAACAAfQBAAB3b3JkL2RvY3VtZW50LnhtbFBLAQIUAxQAAAAIAAAAIQAoRiWwfgAAAJwAAAAcAAAAAAAAAAAAAACAAbwCAAB3b3JkL19yZWxzL2RvY3VtZW50LnhtbC5yZWxzUEsFBgAAAAAEAAQAAwEAAHQDAAAAAA==" // precacted minimal blank docx
                    );
                File.WriteAllBytes(path, bytes);
                return path;
            }
            catch { return null; }
        }

        /// <summary>
        /// 轮询等待我们锁定进程的真正主框架窗口（类名 OpusApp）出现。
        /// 不再依赖 Process.MainWindowHandle（它可能拿到启动画面等过渡窗口）。
        /// </summary>
        private bool WaitForMainWindow()
        {
            int startTick = Environment.TickCount;

            // 辅助门闩：等待消息循环空闲（不作为成功判据）
            try
            {
                if (_wordProcess.WaitForInputIdle(8000))
                    Log("WaitForInputIdle: 消息循环已空闲");
                else
                    Log("WaitForInputIdle: 8s 内未空闲，继续轮询窗口");
            }
            catch (Exception ex) { Log("WaitForInputIdle 异常(忽略): " + ex.Message); }

            int deadline = Environment.TickCount + 30000;
            int round = 0;
            while (Environment.TickCount < deadline)
            {
                round++;
                try { _wordProcess.Refresh(); } catch { }
                if (_wordProcess == null || _wordProcess.HasExited)
                    return Fail("Word 进程已退出，无法嵌入。");

                IntPtr h = FindOpusAppWindow(_pid);
                if (h != IntPtr.Zero)
                {
                    _hwnd = h;
                    Log("取得主框架窗口 HWND=0x" + h.ToString("X") +
                        " class=" + NativeMethods.GetWindowClassName(h) +
                        " 轮次=" + round + " 耗时=" + (Environment.TickCount - startTick) + "ms");
                    return EmbedWindow();
                }

                // 定期输出该 PID 当前所有顶级窗口快照，便于诊断
                if (round == 1 || round % 10 == 0)
                    Log("轮次=" + round + " 未发现 OpusApp。PID 窗口快照: " + DescribePidWindows(_pid));
                Thread.Sleep(200);
            }
            return Fail("等待 Word 主窗口(OpusApp)超时(30s)。最终快照: " + DescribePidWindows(_pid));
        }

        /// <summary>在指定 PID 的可见顶级窗口中查找类名为 OpusApp 的主框架窗口。</summary>
        private IntPtr FindOpusAppWindow(int pid)
        {
            IntPtr withDoc = IntPtr.Zero, first = IntPtr.Zero;
            NativeMethods.EnumWindows((h, l) =>
            {
                try
                {
                    uint wpid;
                    NativeMethods.GetWindowThreadProcessId(h, out wpid);
                    if (wpid != (uint)pid) return true;
                    if (!NativeMethods.IsWindowVisible(h)) return true;
                    if (NativeMethods.GetWindowClassName(h) != "OpusApp") return true;

                    // 已包含文档视图 _WwG 的为最优（Word 完成加载）
                    if (NativeMethods.FindWindowEx(h, IntPtr.Zero, "_WwG", null) != IntPtr.Zero)
                    { withDoc = h; return false; }
                    if (first == IntPtr.Zero) first = h;
                }
                catch { }
                return true;
            }, IntPtr.Zero);
            return withDoc != IntPtr.Zero ? withDoc : first;
        }

        /// <summary>枚举并描述指定 PID 的所有顶级窗口（类名/可见性），用于日志诊断。</summary>
        private static string DescribePidWindows(int pid)
        {
            var list = new List<string>();
            NativeMethods.EnumWindows((h, l) =>
            {
                try
                {
                    uint wpid;
                    NativeMethods.GetWindowThreadProcessId(h, out wpid);
                    if (wpid != (uint)pid) return true;
                    string cls = NativeMethods.GetWindowClassName(h);
                    bool vis = NativeMethods.IsWindowVisible(h);
                    list.Add("0x" + h.ToString("X") + "[" + cls + (vis ? ",可见]" : ",隐藏]"));
                }
                catch { }
                return true;
            }, IntPtr.Zero);
            return list.Count == 0 ? "(无窗口)" : string.Join(" ", list);
        }

        /// <summary>等待 OpusApp 内部出现文档视图子窗口 _WwG（Word 完成加载的信号）。</summary>
        private bool WaitForDocChild(int timeoutMs)
        {
            int deadline = Environment.TickCount + timeoutMs;
            int i = 0;
            while (Environment.TickCount < deadline)
            {
                i++;
                IntPtr doc = NativeMethods.FindWindowEx(_hwnd, IntPtr.Zero, "_WwG", null);
                if (doc != IntPtr.Zero)
                {
                    Log("_WwG 文档子窗口已出现 HWND=0x" + doc.ToString("X") + " 等待次数=" + i);
                    return true;
                }
                Thread.Sleep(200);
            }
            Log("警告: 等待 _WwG 超时(" + timeoutMs + "ms)，继续嵌入流程");
            return false;
        }

        /// <summary>把锁定的主窗口挂靠为本面板子窗口，并铺满、剥壳。</summary>
        private bool EmbedWindow()
        {
            IntPtr h = _hwnd;
            if (h == IntPtr.Zero) return Fail("无效的 Word 窗口句柄。");

            // 嵌入前校验：句柄有效且仍属于我们锁定的进程
            if (!NativeMethods.IsWindow(h))
                return Fail("嵌入前句柄已失效 HWND=0x" + h.ToString("X"));
            uint ownerPid;
            NativeMethods.GetWindowThreadProcessId(h, out ownerPid);
            if (ownerPid != (uint)_pid)
                return Fail("窗口 0x" + h.ToString("X") + " 不属于锁定 PID " + _pid + "（实际 PID=" + ownerPid + "）");

            Log("嵌入前状态: class=" + NativeMethods.GetWindowClassName(h) +
                " style=0x" + NativeMethods.GetWindowStyle(h).ToString("X") +
                " 可见=" + NativeMethods.IsWindowVisible(h) +
                " 面板=" + ClientSize.Width + "x" + ClientSize.Height);

            // 顺序很重要：先隐藏，再改父窗口/样式，最后铺满并显示
            NativeMethods.ShowWindow(h, NativeMethods.SW_HIDE);

            IntPtr prevParent = NativeMethods.SetParent(h, Handle);
            int err = Marshal.GetLastWin32Error();
            Log("SetParent -> 新父=面板 prev=0x" + prevParent.ToString("X") + " GetLastError=" + err);
            if (prevParent == IntPtr.Zero && err != 0)
            {
                NativeMethods.ShowWindow(h, NativeMethods.SW_SHOW); // 恢复可见，避免 Word 凭空消失
                return Fail("SetParent 失败: " + new Win32Exception(err).Message);
            }

            // 去掉标题栏/边框/系统菜单/最小化最大化，改为子窗口
            int oldStyle = NativeMethods.GetWindowStyle(h);
            int style = oldStyle;
            style &= ~(NativeMethods.WS_CAPTION |
                       NativeMethods.WS_THICKFRAME |
                       NativeMethods.WS_SYSMENU |
                       NativeMethods.WS_MINIMIZEBOX |
                       NativeMethods.WS_MAXIMIZEBOX);
            style |= NativeMethods.WS_CHILD | NativeMethods.WS_VISIBLE;
            IntPtr prevStyle = NativeMethods.SetWindowStyle(h, style);
            err = Marshal.GetLastWin32Error();
            Log("SetWindowStyle 0x" + oldStyle.ToString("X") + " -> 0x" + style.ToString("X") +
                " prev=0x" + prevStyle.ToString("X") + " GetLastError=" + err);
            if (prevStyle == IntPtr.Zero && err != 0)
                Log("警告: SetWindowStyle 可能失败: " + new Win32Exception(err).Message);

            // 铺满面板
            bool moved = NativeMethods.MoveWindow(h, 0, 0, Math.Max(1, ClientSize.Width), Math.Max(1, ClientSize.Height), true);
            err = Marshal.GetLastWin32Error();
            Log("MoveWindow " + ClientSize.Width + "x" + ClientSize.Height + " -> " + moved + " GetLastError=" + err);

            // 等待 Word 完成内部加载（_WwG 出现）后再剥壳
            WaitForDocChild(10000);

            // 隐藏 Ribbon 等界面框架（只对我们自己的窗口）
            StripWordChrome();

            // 铺满后显示
            NativeMethods.ShowWindow(h, NativeMethods.SW_SHOW);
            _embedded = true;
            Log("嵌入完成 hwnd=0x" + h.ToString("X") + " size=" + ClientSize.Width + "x" + ClientSize.Height);
            return true;
        }

        /// <summary>
        /// 只枚举我们自己窗口的直接子窗口，隐藏已知 Office 界面框架类
        /// （NUISMDCONTAINER / Mso*），保留文档视图 _WwG 并铺满。
        /// </summary>
        private void StripWordChrome()
        {
            IntPtr doc = IntPtr.Zero;
            IntPtr child = IntPtr.Zero;
            var hidden = new List<string>();
            bool any = false;
            while ((child = NativeMethods.FindWindowEx(_hwnd, child, null, null)) != IntPtr.Zero)
            {
                any = true;
                string cls = NativeMethods.GetWindowClassName(child);
                if (cls == "_WwG")
                {
                    doc = child;
                    continue;
                }
                if (cls.StartsWith("NUISMDCONTAINER") || cls.StartsWith("Mso"))
                {
                    bool ok = NativeMethods.ShowWindow(child, NativeMethods.SW_HIDE);
                    hidden.Add(cls + "(0x" + child.ToString("X") + ",ok=" + ok + ")");
                }
                else
                {
                    Log("子窗口(保留): 0x" + child.ToString("X") + " class=" + cls);
                }
            }
            Log("StripWordChrome: 子窗口" + (any ? "已枚举" : "未发现(Word 内部可能尚未就绪)") +
                " _WwG=" + (doc != IntPtr.Zero ? "0x" + doc.ToString("X") : "未找到") +
                (hidden.Count > 0 ? " 已隐藏: " + string.Join(",", hidden) : " 无匹配隐藏项"));

            if (doc != IntPtr.Zero && ClientSize.Width > 0 && ClientSize.Height > 0)
            {
                bool ok = NativeMethods.MoveWindow(doc, 0, 0, ClientSize.Width, ClientSize.Height, true);
                Log("_WwG 铺满 " + ClientSize.Width + "x" + ClientSize.Height + " -> " + ok);
            }
        }

        /// <summary>把我们自己的 Word 主窗口铺满面板，并同步 _WwG 文档子窗口。</summary>
        private void ResizeToPanel()
        {
            if (_hwnd == IntPtr.Zero || ClientSize.Width <= 0 || ClientSize.Height <= 0) return;
            bool ok = NativeMethods.MoveWindow(_hwnd, 0, 0, ClientSize.Width, ClientSize.Height, true);
            IntPtr doc = NativeMethods.FindWindowEx(_hwnd, IntPtr.Zero, "_WwG", null);
            bool docOk = false;
            if (doc != IntPtr.Zero)
                docOk = NativeMethods.MoveWindow(doc, 0, 0, ClientSize.Width, ClientSize.Height, true);
            Log("ResizeToPanel size=" + ClientSize.Width + "x" + ClientSize.Height +
                " 主窗口ok=" + ok + " _WwG=" + (doc != IntPtr.Zero ? ("0x" + doc.ToString("X") + " ok=" + docOk) : "未找到"));
        }

        /// <summary>本控制器是否仍指向一个有效的、存活的自有进程。</summary>
        private bool IsValid()
        {
            if (_pid == 0 || _hwnd == IntPtr.Zero) return false;
            try
            {
                var proc = Process.GetProcessById(_pid);
                return proc != null && !proc.HasExited && NativeMethods.IsWindow(_hwnd);
            }
            catch { return false; }
        }

        protected override void OnResize(EventArgs e)
        {
            base.OnResize(e);
            if (_embedded && IsValid())
                ResizeToPanel();
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing) Stop();
            base.Dispose(disposing);
        }

        // ==================== 日志/错误 ====================

        private bool Fail(string msg)
        {
            Log("FAIL: " + msg);
            MessageBox.Show(msg, "错误", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return false;
        }

        private static void Log(string msg)
        {
            try
            {
                File.AppendAllText(
                    Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "word_embed_log.txt"),
                    DateTime.Now.ToString("HH:mm:ss.fff") + "  " + msg + Environment.NewLine);
            }
            catch { }
        }
    }
}
