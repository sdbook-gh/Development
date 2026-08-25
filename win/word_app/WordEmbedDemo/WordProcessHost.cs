using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
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
    ///  4) 窗口：MainWindowHandle 从我们锁定的 Process 对象读取，只属于
    ///     这个 PID，绝不会错拿其它 Word 窗口。
    ///  5) 退出：Process.GetProcessById(_pid).Kill() —— 只结束我们自己的
    ///     进程，不影响用户其它 Word。
    ///
    /// 整个过程只用 Win32 API + Process，不写任何 COM 接口，杜绝 vtable
    /// 崩溃（如 0xC0000005）。
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
                SnapshotPreExisting();

                string winword = FindWinWordExe();
                if (winword == null) return Fail("未找到 WINWORD.EXE，请确认已安装 Microsoft Word。");

                // 生成一个最小空白文档，让 Word 打开真实文档窗口（而非“开始页”）
                string blank = CreateBlankDocx();
                if (blank == null) return Fail("无法创建临时空白文档。");

                // /x 强制新实例 + 文档路径：在新实例中打开真实空白文档
                var psi = new ProcessStartInfo(winword, "/x \"" + blank + "\"")
                {
                    UseShellExecute = false,
                    CreateNoWindow = true
                };
                Process p = Process.Start(psi);
                if (p == null) return Fail("启动 Word 失败。");

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

        /// <summary>聚焦我们自己的 Word 窗口并执行 Ctrl+V 粘贴。</summary>
        public void Paste()
        {
            if (!IsValid()) return;
            NativeMethods.SetForegroundWindow(_hwnd);
            NativeMethods.ShowWindow(_hwnd, NativeMethods.SW_RESTORE);
            SendKeys.SendWait("^v");
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

        /// <summary>轮询等待我们锁定进程的主窗口出现。</summary>
        private bool WaitForMainWindow()
        {
            for (int i = 0; i < 75; i++)   // 最多 ~15 秒
            {
                if (_wordProcess == null) return Fail("Word 进程已退出，无法嵌入。");
                try { _wordProcess.Refresh(); }
                catch { return Fail("Word 进程已退出，无法嵌入。"); }
                if (_wordProcess.HasExited)
                    return Fail("Word 进程已退出，无法嵌入。");

                IntPtr h = _wordProcess.MainWindowHandle;
                if (h != IntPtr.Zero)
                {
                    _hwnd = h;
                    Log("取得主窗口 HWND=0x" + h.ToString("X"));
                    return EmbedWindow();
                }
                Thread.Sleep(200);
            }
            return Fail("等待 Word 主窗口超时。");
        }

        /// <summary>把锁定的主窗口挂靠为本面板子窗口，并铺满、剥壳。</summary>
        private bool EmbedWindow()
        {
            IntPtr h = _hwnd;
            if (h == IntPtr.Zero) return Fail("无效的 Word 窗口句柄。");

            // 顺序很重要：先隐藏，再改父窗口/样式，最后铺满并显示，
            // 避免窗口在错误的状态下被 SetParent 导致渲染异常。
            NativeMethods.ShowWindow(h, NativeMethods.SW_HIDE);
            NativeMethods.SetParent(h, Handle);

            // 去掉标题栏/边框/系统菜单/最小化最大化，改为子窗口
            int style = NativeMethods.GetWindowStyle(h);
            style &= ~(NativeMethods.WS_CAPTION |
                       NativeMethods.WS_THICKFRAME |
                       NativeMethods.WS_SYSMENU |
                       NativeMethods.WS_MINIMIZEBOX |
                       NativeMethods.WS_MAXIMIZEBOX);
            style |= NativeMethods.WS_CHILD | NativeMethods.WS_VISIBLE;
            NativeMethods.SetWindowStyle(h, style);

            // 铺满面板
            NativeMethods.MoveWindow(h, 0, 0, Math.Max(1, ClientSize.Width), Math.Max(1, ClientSize.Height), true);

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
            while ((child = NativeMethods.FindWindowEx(_hwnd, child, null, null)) != IntPtr.Zero)
            {
                string cls = NativeMethods.GetWindowClassName(child);
                if (cls == "_WwG")
                {
                    doc = child;
                    continue;
                }
                if (cls.StartsWith("NUISMDCONTAINER") || cls.StartsWith("Mso"))
                    NativeMethods.ShowWindow(child, NativeMethods.SW_HIDE);
            }
            if (doc != IntPtr.Zero && ClientSize.Width > 0 && ClientSize.Height > 0)
                NativeMethods.MoveWindow(doc, 0, 0, ClientSize.Width, ClientSize.Height, true);
        }

        /// <summary>把我们自己的 Word 主窗口铺满面板。</summary>
        private void ResizeToPanel()
        {
            if (_hwnd == IntPtr.Zero || ClientSize.Width <= 0 || ClientSize.Height <= 0) return;
            NativeMethods.MoveWindow(_hwnd, 0, 0, ClientSize.Width, ClientSize.Height, true);
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
