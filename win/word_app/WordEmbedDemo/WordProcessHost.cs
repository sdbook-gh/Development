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
using Microsoft.Win32;
using System.Threading.Tasks;
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
    ///  5) 退出：优先用对象模型优雅退出（ActiveDocument.Close 不保存 +
    ///     Application.Quit），5s 内未退出才回退 Kill() —— 只结束我们自己的
    ///     进程，不影响用户其它 Word，也不留文档恢复面板。
    ///  6) 失败：任何失败出口都走 FailAndCleanup()，立即清理自有进程，
    ///     不留游离（顶层）的 Word 窗口。
    ///
    /// 窗口挂靠只用 Win32 API + Process，不写 OLE 就地嵌入接口（避免 vtable
    /// 崩溃 0xC0000005）。菜单粘贴通过 oleacc 按 HWND 晚期绑定该进程的
    /// Word.Selection，不使用全局 Word.Application。
    /// </summary>
    public class WordProcessHost : Control
    {
        /// <summary>宿主运行状态，供窗体刷新状态栏 / 菜单可用性。</summary>
        public enum HostStatus { Running, Exited, Failed }

        private Process _wordProcess;   // 仅指向我们自己启动的进程
        private int _pid;               // 锁定的进程 PID（0 = 未启动）
        private IntPtr _hwnd;           // 该进程的主窗口句柄
        private bool _embedded;
        private readonly HashSet<int> _preExisting = new HashSet<int>();
        private const int LOG_LIMIT = 1024 * 1024;   // 日志超过 1MB 轮转，避免无限膨胀

        // 防抖：拖动缩放时 resize 事件高频触发，合并到定时器到点后再执行“重排+重剥壳”一次
        private readonly System.Windows.Forms.Timer _resizeDebounce;

        // ---- 生命周期 / 并发防护 ----
        private bool _starting;                     // StartAsync 重入保护
        private bool _disposed;                     // 窗体已销毁后禁止再上抛错误/状态（防幽灵弹框）
        private CancellationTokenSource _cts = new CancellationTokenSource();
        private EventHandler _exitedHandler;        // 命名的 Exited handler，便于 Stop 时解绑

        // 文档链窗口句柄缓存（嵌入后窗口树基本不变，避免每次 resize 递归枚举）
        private IntPtr _cachedWwF, _cachedWwB, _cachedWwG;

        // Word 对象模型常量（替代魔法数字）
        private const int wdDoNotSaveChanges = 0;   // 关闭文档时不保存更改
        private const int wdPrintView = 3;          // 打印版式视图
        private const int wdPageFitFullPage = 1;    // 整页落入窗口

        /// <summary>出错时上抛给宿主窗体，由窗体决定提示方式（本控件不直接弹框）。</summary>
        public event Action<string> HostError;

        /// <summary>宿主状态变化（Running / Exited / Failed），供窗体刷新状态栏与菜单。</summary>
        public event Action<HostStatus> HostStateChanged;

        public WordProcessHost()
        {
            Dock = DockStyle.Fill;
            BackColor = Color.White;

            _resizeDebounce = new System.Windows.Forms.Timer { Interval = 120 };
            _resizeDebounce.Tick += (s, e) =>
            {
                _resizeDebounce.Stop();
                if (_embedded && IsValid())
                    ResizeToPanel();
            };
        }

        // ==================== 公共操作 ====================

        /// <summary>
        /// 启动一个全新的 Word 进程，并挂靠为本面板子窗口。
        /// 异步实现：等待 Word 主窗口/文档视图期间用 await Task.Delay 让出
        /// UI 线程，避免最长 30s 轮询把窗体卡成“未响应”。
        /// </summary>
        public async Task<bool> StartAsync()
        {
            if (_starting)
            {
                Log("StartAsync: 已有启动流程进行中，忽略本次调用");
                return false;
            }
            _starting = true;
            if (_cts != null) _cts.Dispose();
            _cts = new CancellationTokenSource();
            try
            {
                Log("==== 新会话开始 ====");
                SnapshotPreExisting();
                Log("已有 WINWORD 进程快照: " + (_preExisting.Count == 0 ? "(无)" : string.Join(",", _preExisting)));

                string winword = FindWinWordExe();
                Log("WINWORD 路径: " + (winword ?? "(未找到)"));
                if (winword == null) return FailAndCleanup("未找到 WINWORD.EXE，请确认已安装 Microsoft Word。");

                // 生成一个最小空白文档，让 Word 打开真实文档窗口（而非“开始页”）
                string blank = CreateBlankDocx();
                Log("空白文档路径: " + (blank ?? "(创建失败)"));
                if (blank == null) return FailAndCleanup("无法创建临时空白文档。");

                // /x 强制新实例 + 文档路径：在新实例中打开真实空白文档
                var psi = new ProcessStartInfo(winword, "/x \"" + blank + "\"")
                {
                    UseShellExecute = false,
                    CreateNoWindow = true
                };
                Process p = Process.Start(psi);
                if (p == null) return FailAndCleanup("启动 Word 失败。");
                Log("Process.Start 返回 PID=" + p.Id);

                // 防御校验：确保返回的进程确实是“新出现”的，而非被复用的已有实例
                if (_preExisting.Contains(p.Id))
                {
                    try { p.Kill(); } catch { }
                    return FailAndCleanup("Word 复用了已有实例，已放弃，请重试。");
                }

                _wordProcess = p;
                _pid = p.Id;
                Log("锁定新进程 PID=" + _pid);
                WatchProcessExit();

                bool ok = await WaitForMainWindowAsync();
                if (ok)
                    RaiseState(HostStatus.Running);   // 成功启动，通知窗体恢复粘贴等
                return ok;
            }
            catch (Exception ex)
            {
                return FailAndCleanup("启动失败：" + ex.Message);
            }
            finally
            {
                _starting = false;
            }
        }

        /// <summary>监听自有 Word 进程退出（用户在嵌入窗口里 Ctrl+F4/Alt+F4 等），上抛状态变化。</summary>
        private void WatchProcessExit()
        {
            try
            {
                int pid = _pid;
                _exitedHandler = (s, e) =>
                {
                    // 只关心“当前会话”的退出：旧会话被 Stop() 主动关闭时不误报给新会话
                    if (_pid != pid) return;
                    Log("Word 进程已退出 PID=" + pid);
                    RaiseState(HostStatus.Exited);
                };
                _wordProcess.EnableRaisingEvents = true;
                _wordProcess.Exited += _exitedHandler;
            }
            catch (Exception ex) { Log("EnableRaisingEvents 失败(忽略): " + ex.Message); }
        }

        /// <summary>关闭当前进程并重新启动一个全新的 Word（新建文档）。</summary>
        public async Task<bool> NewDocumentAsync()
        {
            Stop();
            return await StartAsync();
        }

        /// <summary>
        /// 菜单粘贴：按已锁定 HWND 取该 Word 进程的对象模型，调用 Selection.Paste()。
        /// 与隔离前相同的 COM 粘贴路径，但不走全局 Word.Application。
        /// </summary>
        public void Paste()
        {
            if (!IsValid()) return;

            object om = null;
            try
            {
                om = BindWordNativeOm();
                if (om == null)
                {
                    Log("Paste FAIL: AccessibleObjectFromWindow 未返回对象");
                    RaiseError("无法绑定到嵌入的 Word 窗口，粘贴失败。");
                    return;
                }

                if (TryPasteSelection(om))
                    return;

                Log("Paste FAIL: 已绑定对象模型，但 Selection.Paste 均失败");
                RaiseError("粘贴失败：Word 未接受 Selection.Paste。");
            }
            catch (Exception ex)
            {
                Log("Paste FAIL: " + ex.GetType().Name + ": " + ex.Message);
                RaiseError("粘贴失败：" + ex.Message);
            }
            finally
            {
                ReleaseCom(om);   // 避免反复粘贴累加 RCW 引用
            }
        }

        /// <summary>对 _WwG（优先）和 OpusApp 调用 AccessibleObjectFromWindow。</summary>
        private object BindWordNativeOm()
        {
            return BindWordNativeOm(_hwnd);
        }

        /// <summary>按指定顶层窗口句柄绑定其所属 Word 进程的对象模型。</summary>
        private object BindWordNativeOm(IntPtr hwnd)
        {
            if (hwnd == IntPtr.Zero) return null;
            IntPtr doc = NativeMethods.FindChildWindowRecursive(hwnd, "_WwG");
            IntPtr[] candidates = doc != IntPtr.Zero
                ? new[] { doc, hwnd }
                : new[] { hwnd };

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
            object sel = null, app = null, win = null;
            try
            {
                try
                {
                    sel = ComGet(om, "Selection");
                    ComCall(sel, "Paste");
                    Log("Paste: Selection.Paste OK");
                    return true;
                }
                catch (Exception ex)
                {
                    Log("Paste: Selection.Paste: " + ex.Message);
                }

                try
                {
                    app = ComGet(om, "Application");
                    ReleaseCom(sel); sel = null;
                    sel = ComGet(app, "Selection");
                    ComCall(sel, "Paste");
                    Log("Paste: Application.Selection.Paste OK");
                    return true;
                }
                catch (Exception ex)
                {
                    Log("Paste: Application.Selection.Paste: " + ex.Message);
                }

                try
                {
                    win = ComGet(om, "ActiveWindow");
                    ReleaseCom(sel); sel = null;
                    sel = ComGet(win, "Selection");
                    ComCall(sel, "Paste");
                    Log("Paste: ActiveWindow.Selection.Paste OK");
                    return true;
                }
                catch (Exception ex)
                {
                    Log("Paste: ActiveWindow.Selection.Paste: " + ex.Message);
                }

                return false;
            }
            finally
            {
                ReleaseCom(sel); ReleaseCom(app); ReleaseCom(win);
            }
        }

        /// <summary>释放 COM 对象的 RCW 引用（非 COM 对象/空值静默忽略）。</summary>
        private static void ReleaseCom(object o)
        {
            try
            {
                if (o != null && Marshal.IsComObject(o))
                    Marshal.ReleaseComObject(o);
            }
            catch { }
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

        private static void ComSet(object target, string name, object value)
        {
            target.GetType().InvokeMember(name,
                BindingFlags.SetProperty | BindingFlags.Instance, null, target, new object[] { value });
        }


        /// <summary>在 OpusApp 后代窗口中查找可见的文档视图 _WwG。</summary>
        /// <remarks>
        /// 必须递归查找：Word 真实层级是 OpusApp &gt; _WwF &gt; _WwB &gt; _WwG，
        /// 而 FindWindowEx 只查直接子窗口，永远找不到 _WwG。
        /// </remarks>
        private IntPtr FindDocView()
        {
            return FindDocView(_hwnd);
        }

        private IntPtr FindDocView(IntPtr hwnd)
        {
            if (hwnd == IntPtr.Zero) return IntPtr.Zero;
            return NativeMethods.FindChildWindowRecursive(hwnd, "_WwG");
        }

        /// <summary>只结束我们自己的 Word 进程（不影响用户其它 Word）。默认后台优雅退出，不阻塞调用线程。</summary>
        public void Stop()
        {
            Stop(forceKill: false);
        }

        /// <summary>
        /// 结束自有 Word 进程。
        /// forceKill=true：立即 Kill（窗体关闭等需立即释放的场景，不做优雅退出，避免阻塞 UI）；
        /// forceKill=false：后台优雅退出（ActiveDocument.Close 不保存 + Application.Quit），
        /// 不阻塞调用线程，避免关窗 / 新建文档时界面卡死。
        /// </summary>
        public void Stop(bool forceKill)
        {
            int pid = _pid;
            IntPtr hwnd = _hwnd;
            var cts = _cts;
            if (cts != null) cts.Cancel();

            if (pid != 0 && _wordProcess != null)
            {
                // 先解绑退出事件，避免 Stop 后旧进程 Exited 再触发状态变化
                try
                {
                    if (_exitedHandler != null) _wordProcess.Exited -= _exitedHandler;
                    _wordProcess.EnableRaisingEvents = false;
                }
                catch (Exception ex) { Log("解绑 Exited 事件(忽略): " + ex.Message); }

                if (forceKill)
                    KillProcess(pid);
                else
                    Task.Run(() => TryCloseGracefully(pid, hwnd));
            }

            _pid = 0;
            _wordProcess = null;
            _hwnd = IntPtr.Zero;
            _embedded = false;
            _cachedWwF = _cachedWwB = _cachedWwG = IntPtr.Zero;
        }

        /// <summary>立即结束指定进程（只结束我们自己的，不影响用户其它 Word）。</summary>
        private static void KillProcess(int pid)
        {
            try
            {
                using (var proc = Process.GetProcessById(pid))
                {
                    if (!proc.HasExited)
                    {
                        proc.Kill();
                        Log("退出方式=kill (PID=" + pid + ")");
                    }
                }
            }
            catch (Exception ex) { Log("kill 异常(忽略): " + ex.Message); }
        }

        /// <summary>指定 PID 的进程是否仍然存活。</summary>
        private static bool ProcessAlive(int pid)
        {
            try
            {
                using (var proc = Process.GetProcessById(pid))
                    return !proc.HasExited;
            }
            catch { return false; }
        }

        /// <summary>
        /// 在后台线程执行：优先用对象模型优雅退出 ActiveDocument.Close(不保存) + Application.Quit，
        /// 最多等 5s，未退出则回退 Kill()。
        /// </summary>
        private void TryCloseGracefully(int pid, IntPtr hwnd)
        {
            if (pid == 0) return;

            object om = null;
            try
            {
                if (hwnd != IntPtr.Zero && NativeMethods.IsWindow(hwnd))
                    om = BindWordNativeOm(hwnd);

                if (om != null)
                {
                    object app = ComGet(om, "Application");
                    try
                    {
                        try
                        {
                            object doc = ComGet(app, "ActiveDocument");
                            if (doc != null)
                            {
                                // wdDoNotSaveChanges：不弹“是否保存”，也不丢给下次恢复
                                doc.GetType().InvokeMember("Close",
                                    BindingFlags.InvokeMethod | BindingFlags.Instance,
                                    null, doc, new object[] { wdDoNotSaveChanges });
                                Log("graceful: ActiveDocument.Close(不保存) OK");
                                ReleaseCom(doc);
                            }
                        }
                        catch (Exception ex) { Log("graceful: ActiveDocument.Close: " + ex.Message); }

                        app.GetType().InvokeMember("Quit",
                            BindingFlags.InvokeMethod | BindingFlags.Instance, null, app, null);
                        Log("graceful: Application.Quit 已发送");
                    }
                    finally { ReleaseCom(app); }
                }
                else
                {
                    Log("graceful: 未绑定到对象模型，直接转 Kill");
                }
            }
            catch (Exception ex) { Log("graceful: 退出异常(转 Kill): " + ex.GetType().Name + ": " + ex.Message); }
            finally { ReleaseCom(om); }

            // 最多等 5s，未退出则回退 Kill
            var deadline = DateTime.UtcNow.AddSeconds(5);
            while (DateTime.UtcNow < deadline)
            {
                if (!ProcessAlive(pid)) break;
                Thread.Sleep(100);
            }

            if (ProcessAlive(pid))
            {
                Log("graceful: 5s 内未退出，回退 Kill");
                KillProcess(pid);
            }
            else
            {
                Log("退出方式=graceful (PID=" + pid + ")");
            }
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
            // 优先从注册表 App Paths 探测（覆盖 2013/2016/2019/365 任意安装路径与自定义安装）
            string fromRegistry = FindWinWordFromRegistry();
            if (fromRegistry != null) return fromRegistry;

            string[] candidates =
            {
                @"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
                @"C:\Program Files (x86)\Microsoft Office\root\Office16\WINWORD.EXE",
                @"C:\Program Files\Microsoft Office\Office16\WINWORD.EXE",
                @"C:\Program Files (x86)\Microsoft Office\Office16\WINWORD.EXE",
                @"C:\Program Files\Microsoft Office\root\Office15\WINWORD.EXE",
                @"C:\Program Files (x86)\Microsoft Office\root\Office15\WINWORD.EXE",
            };
            foreach (var c in candidates)
                if (File.Exists(c)) return c;

            // 兜底：从标准安装路径探测
            try
            {
                string office = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86);
                string p = Path.Combine(office, @"Microsoft Office\root\Office16\WINWORD.EXE");
                if (File.Exists(p)) return p;
            }
            catch { }
            return null;
        }

        /// <summary>从注册表 App Paths 读取 WINWORD.EXE 完整路径（最可靠的探测方式）。</summary>
        private static string FindWinWordFromRegistry()
        {
            string[] subKeys =
            {
                @"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\winword.exe",
                @"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths\winword.exe",
            };
            try
            {
                foreach (var sub in subKeys)
                {
                    using (var key = Registry.LocalMachine.OpenSubKey(sub))
                    {
                        if (key != null)
                        {
                            string v = key.GetValue(null) as string;   // (Default) 值即完整路径
                            if (!string.IsNullOrEmpty(v) && File.Exists(v)) return v;
                        }
                    }
                }
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

                // 先清理历史遗留的空白文档：固定文件名会因上个进程未释放而
                // 触发 Word 的“文件正在使用中 / 只读”模态框，阻塞嵌入
                try
                {
                    foreach (var old in Directory.GetFiles(dir, "blank-*.docx"))
                    {
                        try { File.Delete(old); } catch { }   // 仍被占用则忽略
                    }
                }
                catch { }

                string path = Path.Combine(dir, "blank-" + Guid.NewGuid().ToString("N") + ".docx");
                byte[] bytes = Convert.FromBase64String(
                    "UEsDBBQAAAAIAAAAIQAXmADX6wAAALIBAAATAAAAW0NvbnRlbnRfVHlwZXNdLnhtbH1QyU4DMQy98xWRr2gmAweEUKc9sByBQ/kAK/HMRM2mOC3t3+NpoQdUONpvs99itQ9e7aiwS7GHm7YDRdEk6+LYw8f6pbkHxRWjRZ8i9XAghtXyarE+ZGIl4sg9TLXmB63ZTBSQ25QpCjKkErDKWEad0WxwJH3bdXfapFgp1qbOHiBmTzTg1lf1vJf96ZJCnkE9nphzWA+Ys3cGq+B6F+2vmOY7ohXlkcOTy3wtBNCXI2bo74Qf4ZuUU5wl9Y6lvmIQmv5MxWqbzDaItP3f58KlaRicobN+dsslGWKW1oNvz0hAF88f6GPlyy9QSwMEFAAAAAgAAAAhAD+t/vqvAAAALAEAAAsAAABfcmVscy8ucmVsc43POw7CMAwA0J1TRN5pWgaEUEMXhNQVlQNEiZtWNB/F4dPbk4EBKgZG/57tunnaid0x0uidgKoogaFTXo/OCLh0p/UOGCXptJy8QwEzEjSHVX3GSaY8Q8MYiGXEkYAhpbDnnNSAVlLhA7pc6X20MuUwGh6kukqDfFOWWx4/DVigrNUCYqsrYN0c8B/c9/2o8OjVzaJLP3YsOrIso8Ek4OGj5vqdLjILPJ/Dv548vABQSwMEFAAAAAgAAAAhAIWE7cmZAAAAywAAABEAAAB3b3JkL2RvY3VtZW50LnhtbEWOwQ7CIBBE734F2bulejCmKfTmF+gHIGBtUnYJi9b+vVAPXt5kM5OZ7YdPmMXbJ54IFRyaFoRHS27CUcHtetmfQXA26MxM6BWsnmHQu37pHNlX8JhFaUDuFgXPnGMnJdunD4Ybih6L96AUTC5nGuVCycVE1jOXgTDLY9ueZDATgi6Vd3Jr1ViRKrIWvaxSmTbGjb+o/L+hv1BLAwQUAAAACAAAACEAKEYlsH4AAACcAAAAHAAAAHdvcmQvX3JlbHMvZG9jdW1lbnQueG1sLnJlbHNVzEEOwiAQheG9pyCzt6ALYwy0ux7A6AEmdARiOxCGGL29LHX58ud9dnpvq3pRlZTZwWEwoIh9XhIHB/fbvD+Dkoa84JqZHHxIYBp39kortv6RmIqojrA4iK2Vi9biI20oQy7EvTxy3bD1WYMu6J8YSB+NOen6a8Bo9R86fgFQSwECFAMUAAAACAAAACEAF5gA1+sAAACyAQAAEwAAAAAAAAAAAAAAgAEAAAAAW0NvbnRlbnRfVHlwZXNdLnhtbFBLAQIUAxQAAAAIAAAAIQA/rf76rwAAACwBAAALAAAAAAAAAAAAAACAARwBAABfcmVscy8ucmVsc1BLAQIUAxQAAAAIAAAAIQCFhO3JmQAAAMsAAAARAAAAAAAAAAAAAACAAfQBAAB3b3JkL2RvY3VtZW50LnhtbFBLAQIUAxQAAAAIAAAAIQAoRiWwfgAAAJwAAAAcAAAAAAAAAAAAAACAAbwCAAB3b3JkL19yZWxzL2RvY3VtZW50LnhtbC5yZWxzUEsFBgAAAAAEAAQAAwEAAHQDAAAAAA==" // precacted minimal blank docx
                    );
                // 内嵌 docx 自检：数据被误改/截断时给出明确诊断，而不是让 Word 报“文件已损坏”
                if (bytes == null || bytes.Length < 1024)
                {
                    Log("FAIL: 内嵌空白 docx 数据异常(len=" + (bytes == null ? -1 : bytes.Length) + ")");
                    return null;
                }
                File.WriteAllBytes(path, bytes);
                return path;
            }
            catch (Exception ex)
            {
                Log("FAIL: 创建空白 docx 异常: " + ex.GetType().Name + ": " + ex.Message);
                return null;
            }
        }

        /// <summary>
        /// 轮询等待我们锁定进程的真正主框架窗口（类名 OpusApp）出现。
        /// 不再依赖 Process.MainWindowHandle（它可能拿到启动画面等过渡窗口）。
        /// </summary>
        private async Task<bool> WaitForMainWindowAsync()
        {
            var startTime = DateTime.UtcNow;

            // 辅助门闩：等待消息循环空闲（不作为成功判据）——放到后台线程，避免阻塞 UI
            try
            {
                bool idle = await Task.Run(() => _wordProcess.WaitForInputIdle(8000));
                if (idle)
                    Log("WaitForInputIdle: 消息循环已空闲");
                else
                    Log("WaitForInputIdle: 8s 内未空闲，继续轮询窗口");
            }
            catch (Exception ex) { Log("WaitForInputIdle 异常(忽略): " + ex.Message); }

            var deadline = DateTime.UtcNow.AddSeconds(30);
            int round = 0;
            while (DateTime.UtcNow < deadline)
            {
                if (_cts.Token.IsCancellationRequested)   // Stop 已触发，静默放弃
                    return false;
                round++;
                if (_wordProcess == null || _wordProcess.HasExited)
                    return FailAndCleanup("Word 进程已退出，无法嵌入。");
                try { _wordProcess.Refresh(); } catch { }

                IntPtr h = FindOpusAppWindow(_pid);
                if (h != IntPtr.Zero)
                {
                    _hwnd = h;
                    Log("取得主框架窗口 HWND=0x" + h.ToString("X") +
                        " class=" + NativeMethods.GetWindowClassName(h) +
                        " 轮次=" + round + " 耗时=" + (DateTime.UtcNow - startTime).TotalMilliseconds + "ms");
                    return await EmbedWindowAsync();
                }

                // 定期输出该 PID 当前所有顶级窗口快照，便于诊断
                if (round == 1 || round % 10 == 0)
                    Log("轮次=" + round + " 未发现 OpusApp。PID 窗口快照: " + DescribePidWindows(_pid));
                await Task.Delay(200);
            }
            return FailAndCleanup("等待 Word 主窗口(OpusApp)超时(30s)。最终快照: " + DescribePidWindows(_pid));
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
                    // 必须递归查：_WwG 在 OpusApp &gt; _WwF &gt; _WwB 下，不是直接子窗口
                    if (NativeMethods.FindChildWindowRecursive(h, "_WwG") != IntPtr.Zero)
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
        private async Task<bool> WaitForDocChildAsync(int timeoutMs)
        {
            var deadline = DateTime.UtcNow.AddMilliseconds(timeoutMs);
            int i = 0;
            while (DateTime.UtcNow < deadline)
            {
                if (_cts.Token.IsCancellationRequested) return false;
                i++;
                IntPtr doc = NativeMethods.FindChildWindowRecursive(_hwnd, "_WwG");
                if (doc != IntPtr.Zero)
                {
                    Log("_WwG 文档子窗口已出现 HWND=0x" + doc.ToString("X") + " 等待次数=" + i);
                    return true;
                }
                await Task.Delay(200);
            }
            Log("警告: 等待 _WwG 超时(" + timeoutMs + "ms)，继续嵌入流程");
            return false;
        }

        /// <summary>把锁定的主窗口挂靠为本面板子窗口，并铺满、剥壳。</summary>
        private async Task<bool> EmbedWindowAsync()
        {
            IntPtr h = _hwnd;
            if (h == IntPtr.Zero) return FailAndCleanup("无效的 Word 窗口句柄。");

            // 嵌入前校验：句柄有效且仍属于我们锁定的进程
            if (!NativeMethods.IsWindow(h))
                return FailAndCleanup("嵌入前句柄已失效 HWND=0x" + h.ToString("X"));
            uint ownerPid;
            NativeMethods.GetWindowThreadProcessId(h, out ownerPid);
            if (ownerPid != (uint)_pid)
                return FailAndCleanup("窗口 0x" + h.ToString("X") + " 不属于锁定 PID " + _pid + "（实际 PID=" + ownerPid + "）");

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
                // 失败常见原因：Word 以更高权限运行（UIPI）会直接拒绝跨进程 SetParent
                NativeMethods.ShowWindow(h, NativeMethods.SW_SHOW); // 先恢复可见，再清理
                return FailAndCleanup("SetParent 失败: " + new Win32Exception(err).Message +
                    "\n若 Word / Office 服务以管理员权限运行，请与本程序保持相同权限级别。");
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

            // 样式变更后必须通知框架重算非客户区，否则 Word 会按旧标题高度排版（与剥壳/铺满对抗）
            bool frameOk = NativeMethods.SetWindowPos(h, IntPtr.Zero, 0, 0, 0, 0,
                NativeMethods.SWP_FRAMECHANGED | NativeMethods.SWP_NOMOVE |
                NativeMethods.SWP_NOSIZE | NativeMethods.SWP_NOZORDER);
            Log("SetWindowPos(FRAMECHANGED) -> " + frameOk + " GetLastError=" + Marshal.GetLastWin32Error());

            // 铺满面板
            bool moved = NativeMethods.MoveWindow(h, 0, 0, Math.Max(1, ClientSize.Width), Math.Max(1, ClientSize.Height), true);
            err = Marshal.GetLastWin32Error();
            Log("MoveWindow " + ClientSize.Width + "x" + ClientSize.Height + " -> " + moved + " GetLastError=" + err);

            // 等待 Word 完成内部加载（_WwG 出现）后再剥壳
            await WaitForDocChildAsync(10000);
            if (_cts.Token.IsCancellationRequested)   // 等待期间被 Stop，放弃嵌入
                return false;

            RelayoutEmbeddedWord();

            // 铺满后显示
            NativeMethods.ShowWindow(h, NativeMethods.SW_SHOW);
            _embedded = true;
            Log("嵌入完成 hwnd=0x" + h.ToString("X") + " size=" + ClientSize.Width + "x" + ClientSize.Height);
            return true;
        }

        /// <summary>
        /// 嵌入后统一重排：清零 chrome 占位，按 OpusApp → _WwF → _WwB → _WwG 铺满面板，
        /// 再用对象模型关标尺/任务窗格并把纸张 PageFit 居中。
        /// Embed 与 Resize 共用，避免拖拽后又偏回去。
        /// </summary>
        private void RelayoutEmbeddedWord()
        {
            if (_hwnd == IntPtr.Zero || ClientSize.Width <= 0 || ClientSize.Height <= 0) return;

            NativeMethods.SendMessage(_hwnd, NativeMethods.WM_SETREDRAW, (IntPtr)0, IntPtr.Zero);
            try
            {
                StripWordChrome();
                FillDocumentChain();
                StripChromeViaOm();
                // OM 改 Zoom/标尺会触发布局，再剥一次并铺满
                StripWordChrome();
                FillDocumentChain();
            }
            finally
            {
                NativeMethods.SendMessage(_hwnd, NativeMethods.WM_SETREDRAW, (IntPtr)1, IntPtr.Zero);
                NativeMethods.InvalidateRect(_hwnd, IntPtr.Zero, true);
            }

            Log("RelayoutEmbeddedWord size=" + ClientSize.Width + "x" + ClientSize.Height);
        }

        /// <summary>
        /// 只枚举我们自己窗口的直接子窗口，隐藏已知 Office 界面框架
        /// （NUISMDCONTAINER / Mso* / NetUIHWND），并把它们尺寸清零，避免仍占布局。
        /// </summary>
        private void StripWordChrome()
        {
            IntPtr child = IntPtr.Zero;
            var hidden = new List<string>();
            bool any = false;
            while ((child = NativeMethods.FindWindowEx(_hwnd, child, null, null)) != IntPtr.Zero)
            {
                any = true;
                string cls = NativeMethods.GetWindowClassName(child);
                if (cls == "_WwF" || cls == "_WwB" || cls == "_WwG")
                    continue;
                if (cls.StartsWith("NUISMDCONTAINER") || cls.StartsWith("Mso") || cls == "NetUIHWND")
                {
                    bool ok = NativeMethods.ShowWindow(child, NativeMethods.SW_HIDE);
                    int childStyle = NativeMethods.GetWindowStyle(child);
                    childStyle &= ~NativeMethods.WS_VISIBLE;
                    NativeMethods.SetWindowStyle(child, childStyle);
                    NativeMethods.MoveWindow(child, 0, 0, 1, 1, true);
                    hidden.Add(cls + "(0x" + child.ToString("X") + ",ok=" + ok + ")");
                }
            }
            // resize 高频路径不逐个子窗口记日志，仅在有实质变化时输出
            if (!any)
                Log("StripWordChrome: 未发现子窗口(Word 内部可能尚未就绪)");
            else if (hidden.Count > 0)
                Log("StripWordChrome: 已隐藏 " + string.Join(",", hidden));
        }

        /// <summary>
        /// 按 OpusApp → _WwF → _WwB → _WwG 把文档区铺满宿主面板。
        /// MoveWindow 坐标相对父窗口，必须一层层铺，不能只动 _WwG。
        /// </summary>
        private void FillDocumentChain()
        {
            int w = Math.Max(1, ClientSize.Width);
            int h = Math.Max(1, ClientSize.Height);

            bool okApp = NativeMethods.MoveWindow(_hwnd, 0, 0, w, h, true);

            // 窗口树在嵌入后基本不变，缓存句柄避免每次 resize 都递归枚举
            IntPtr frame = GetDocChainHandle(ref _cachedWwF, "_WwF");
            IntPtr border = GetDocChainHandle(ref _cachedWwB, "_WwB");
            IntPtr doc = GetDocChainHandle(ref _cachedWwG, "_WwG");

            bool okF = false;
            if (frame != IntPtr.Zero)
                okF = NativeMethods.MoveWindow(frame, 0, 0, w, h, true);
            bool okB = false;
            if (border != IntPtr.Zero)
                okB = NativeMethods.MoveWindow(border, 0, 0, w, h, true);
            bool okG = false;
            if (doc != IntPtr.Zero)
                okG = NativeMethods.MoveWindow(doc, 0, 0, w, h, true);

            // resize 高频路径只记概要，避免日志刷屏
            Log("FillDocumentChain " + w + "x" + h +
                " OpusApp=" + okApp +
                " _WwF=" + (frame != IntPtr.Zero ? okF.ToString() : "未找到") +
                " _WwB=" + (border != IntPtr.Zero ? okB.ToString() : "未找到") +
                " _WwG=" + (doc != IntPtr.Zero ? okG.ToString() : "未找到"));
        }

        /// <summary>取文档链窗口句柄：缓存命中且仍有效则直接复用，失效才递归重查并刷新缓存。</summary>
        private IntPtr GetDocChainHandle(ref IntPtr cache, string className)
        {
            if (cache != IntPtr.Zero && NativeMethods.IsWindow(cache))
                return cache;
            IntPtr h = NativeMethods.FindWindowEx(_hwnd, IntPtr.Zero, className, null);
            if (h == IntPtr.Zero)
                h = NativeMethods.FindChildWindowRecursive(_hwnd, className);
            cache = h;
            return h;
        }

        /// <summary>
        /// 剥壳叠加层：关 Ribbon / 状态栏 / 标尺 / 任务窗格，并把纸张 PageFit 居中。
        /// 失败降级：任何异常只记日志，不影响几何铺满。
        /// </summary>
        private void StripChromeViaOm()
        {
            object om = null;
            try
            {
                om = BindWordNativeOm();
                if (om == null)
                {
                    Log("StripChromeViaOm: 未绑定对象模型，跳过（仅依赖几何剥壳）");
                    return;
                }

                object app = ComGet(om, "Application");
                try
                {
                    try
                    {
                        object cbs = ComGet(app, "CommandBars");
                        try
                        {
                            TrySetCommandBarEnabled(cbs, "Ribbon", false);
                            TrySetCommandBarVisible(cbs, "Task Pane", false);
                            TrySetCommandBarVisible(cbs, "Navigation", false);
                        }
                        finally { ReleaseCom(cbs); }
                    }
                    catch (Exception ex) { Log("StripChromeViaOm: CommandBars: " + ex.Message); }

                    try
                    {
                        ComSet(app, "DisplayStatusBar", false);
                        Log("StripChromeViaOm: DisplayStatusBar=false OK");
                    }
                    catch (Exception ex) { Log("StripChromeViaOm: DisplayStatusBar: " + ex.Message); }

                    object win = null;
                    try
                    {
                        win = ComGet(app, "ActiveWindow");
                        try { ComSet(win, "DisplayRulers", false); Log("StripChromeViaOm: DisplayRulers=false OK"); }
                        catch (Exception ex) { Log("StripChromeViaOm: DisplayRulers: " + ex.Message); }
                        try { ComSet(win, "DocumentMap", false); Log("StripChromeViaOm: DocumentMap=false OK"); }
                        catch (Exception ex) { Log("StripChromeViaOm: DocumentMap: " + ex.Message); }

                        object view = null, zoom = null;
                        try
                        {
                            view = ComGet(win, "View");
                            try { ComSet(view, "Type", wdPrintView); Log("StripChromeViaOm: View.Type=wdPrintView OK"); }
                            catch (Exception ex) { Log("StripChromeViaOm: View.Type: " + ex.Message); }
                            zoom = ComGet(view, "Zoom");
                            // wdPageFitFullPage：整页落入窗口，Word 会把纸张放在视图正中
                            ComSet(zoom, "PageFit", wdPageFitFullPage);
                            Log("StripChromeViaOm: Zoom.PageFit=wdPageFitFullPage OK");
                        }
                        catch (Exception ex) { Log("StripChromeViaOm: Zoom.PageFit: " + ex.Message); }
                        finally { ReleaseCom(zoom); ReleaseCom(view); }
                    }
                    catch (Exception ex) { Log("StripChromeViaOm: ActiveWindow: " + ex.Message); }
                    finally { ReleaseCom(win); }
                }
                finally { ReleaseCom(app); }
            }
            catch (Exception ex)
            {
                Log("StripChromeViaOm: 降级(忽略) " + ex.GetType().Name + ": " + ex.Message);
            }
            finally { ReleaseCom(om); }
        }

        private static void TrySetCommandBarEnabled(object commandBars, string name, bool enabled)
        {
            object bar = null;
            try
            {
                bar = commandBars.GetType().InvokeMember("Item",
                    BindingFlags.GetProperty | BindingFlags.Instance,
                    null, commandBars, new object[] { name });
                ComSet(bar, "Enabled", enabled);
                Log("StripChromeViaOm: CommandBars(" + name + ").Enabled=" + enabled + " OK");
            }
            catch (Exception ex) { Log("StripChromeViaOm: CommandBars(" + name + ").Enabled: " + ex.Message); }
            finally { ReleaseCom(bar); }
        }

        private static void TrySetCommandBarVisible(object commandBars, string name, bool visible)
        {
            object bar = null;
            try
            {
                bar = commandBars.GetType().InvokeMember("Item",
                    BindingFlags.GetProperty | BindingFlags.Instance,
                    null, commandBars, new object[] { name });
                ComSet(bar, "Visible", visible);
                Log("StripChromeViaOm: CommandBars(" + name + ").Visible=" + visible + " OK");
            }
            catch (Exception ex) { Log("StripChromeViaOm: CommandBars(" + name + ").Visible: " + ex.Message); }
            finally { ReleaseCom(bar); }
        }

        /// <summary>把我们自己的 Word 主窗口铺满面板，并同步文档子窗口与纸张居中。</summary>
        /// <remarks>
        /// 调整大小会触发 Word 重新布局内部 UI，被隐藏的菜单栏/工具栏
        /// 会被它重新显示出来，因此必须走 RelayoutEmbeddedWord（重铺 + 重剥壳）。
        /// </remarks>
        private void ResizeToPanel()
        {
            RelayoutEmbeddedWord();
        }

        /// <summary>本控制器是否仍指向一个有效的、存活的自有进程。</summary>
        private bool IsValid()
        {
            if (_pid == 0 || _hwnd == IntPtr.Zero) return false;
            try
            {
                // HasExited 是 Process 缓存的属性，不抛异常、不枚举进程列表，比 GetProcessById 轻量
                return _wordProcess != null && !_wordProcess.HasExited && NativeMethods.IsWindow(_hwnd);
            }
            catch { return false; }
        }

        protected override void OnResize(EventArgs e)
        {
            base.OnResize(e);
            if (_disposed) return;
            if (_embedded && IsValid())
            {
                // 防抖：合并 resize 期间高频事件，120ms 后只执行一次“重排+重剥壳”
                _resizeDebounce.Stop();
                _resizeDebounce.Start();
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _disposed = true;
                try { if (_cts != null) _cts.Cancel(); } catch { }
                try { if (_cts != null) _cts.Dispose(); } catch { }
                if (_resizeDebounce != null)
                {
                    _resizeDebounce.Stop();
                    _resizeDebounce.Dispose();
                }
                Stop(forceKill: true);
            }
            base.Dispose(disposing);
        }

        // ==================== 日志/错误 ====================

        private bool Fail(string msg)
        {
            Log("FAIL: " + msg);
            RaiseError(msg);   // 由宿主窗体决定如何提示，控件不直接弹框
            RaiseState(HostStatus.Failed);
            return false;
        }

        /// <summary>失败出口统一入口：先清理自有进程，避免遗留游离的顶层 Word 窗口。</summary>
        private bool FailAndCleanup(string msg)
        {
            Stop();
            return Fail(msg);
        }

        /// <summary>上抛错误给宿主窗体（约定在 UI 线程调用）。窗体已销毁时静默丢弃，避免幽灵弹框。</summary>
        private void RaiseError(string msg)
        {
            if (_disposed) return;
            var h = HostError;
            if (h != null) h(msg);
        }

        /// <summary>上抛状态变化；Process.Exited 在线程池线程触发，需切回 UI 线程。窗体已销毁时静默丢弃。</summary>
        private void RaiseState(HostStatus status)
        {
            if (_disposed) return;
            var h = HostStateChanged;
            if (h == null) return;
            try
            {
                if (IsHandleCreated)
                    BeginInvoke(new Action(() => h(status)));
                else
                    h(status);
            }
            catch { }
        }

        /// <summary>统一日志入口（窗体全局异常处理也写这里，保证单一时间线）。超 1MB 轮转一次。</summary>
        private static readonly object _logLock = new object();

        internal static void Log(string msg)
        {
            try
            {
                // UI 线程 / 线程池(Exited) / async 续体并发写日志，必须加锁防互相踩踏
                lock (_logLock)
                {
                    string file = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "word_embed_log.txt");
                    try
                    {
                        var fi = new FileInfo(file);
                        if (fi.Exists && fi.Length > LOG_LIMIT)
                        {
                            if (File.Exists(file + ".1")) File.Delete(file + ".1");
                            File.Move(file, file + ".1");
                        }
                    }
                    catch { }
                    File.AppendAllText(file,
                        DateTime.Now.ToString("HH:mm:ss.fff") + "  " + msg + Environment.NewLine);
                }
            }
            catch { }
        }
    }
}
