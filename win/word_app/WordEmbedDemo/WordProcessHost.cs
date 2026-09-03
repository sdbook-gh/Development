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
    ///  5) 退出：先在 UI/STA 上还原剥壳前保存的 Word 界面属性（Ribbon /
    ///     状态栏 / 标尺等），再 ActiveDocument.Close + Application.Quit，
    ///     短超时未退出才 Kill() —— 避免把用户自己的 Word 剥成无 Ribbon。
    ///  6) 失败：任何失败出口都走 FailAndCleanup()，强制结束并等待自有进程
    ///     退出，不 fire-and-forget 优雅 Quit，不留第二份 WINWORD。
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
        private bool _stopping;                    // Stop 期间禁止 Relayout/剥壳
        private string _blankDocxPath;             // 本次会话空白文档，退出后删除
        private CancellationTokenSource _cts = new CancellationTokenSource();
        private EventHandler _exitedHandler;        // 命名的 Exited handler，便于 Stop 时解绑

        // 文档链窗口句柄缓存（嵌入后窗口树基本不变，避免每次 resize 递归枚举）
        private IntPtr _cachedWwF, _cachedWwB, _cachedWwG;

        // Word 对象模型常量（替代魔法数字）
        private const int wdDoNotSaveChanges = 0;   // 关闭文档时不保存更改
        private const int wdPrintView = 3;          // 打印版式视图
        private const int wdPageFitFullPage = 1;    // 整页落入窗口
        private const ushort IMAGE_FILE_MACHINE_I386 = 0x14C;
        private const ushort IMAGE_FILE_MACHINE_AMD64 = 0x8664;
        private const int GRACEFUL_QUIT_TIMEOUT_MS = 5000;
        private const int FORCE_KILL_WAIT_MS = 2000;

        /// <summary>第一次剥壳前保存的 Application/窗口界面属性，退出时还原。</summary>
        private readonly ChromeSnapshot _chrome = new ChromeSnapshot();

        private sealed class ChromeSnapshot
        {
            public bool Captured;
            public object RibbonEnabled;
            public object TaskPaneVisible;
            public object NavigationVisible;
            public object DisplayStatusBar;
            public object DisplayRulers;
            public object DocumentMap;
            public object ViewType;
            public object PageFit;
        }

        /// <summary>出错时上抛给宿主窗体，由窗体决定提示方式（本控件不直接弹框）。</summary>
        public event Action<string> HostError;
        /// <summary>最近一次失败原因，供窗体在 StartAsync 返回 false 时弹框。</summary>
        public string LastError { get; private set; }
        /// <summary>取消启动时 LastError 标记；窗体不弹「嵌入失败」。</summary>
        public const string CancelledSentinel = "CANCELLED";


        /// <summary>宿主状态变化（Running / Exited / Failed），供窗体刷新状态栏与菜单。</summary>
        public event Action<HostStatus> HostStateChanged;

        public WordProcessHost()
        {
            // 不再使用 Dock=Fill 铺满窗体，由 MainForm 控制居中布局
            BackColor = Color.White;

            _resizeDebounce = new System.Windows.Forms.Timer { Interval = 120 };
            _resizeDebounce.Tick += (s, e) =>
            {
                _resizeDebounce.Stop();
                if (!_stopping && _embedded && IsValid())
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
                // 禁止第二份 WINWORD：若上一次宿主仍在，先强制结束并等待
                if (_pid != 0 || IsHostProcessAlive())
                {
                    Log("StartAsync: 发现仍在运行的宿主 Word (PID=" + _pid + ")，先强制结束");
                    EnsureHostedWordStopped(true);
                }
                _stopping = false;
                SnapshotPreExisting();
                Log("已有 WINWORD 进程快照: " + (_preExisting.Count == 0 ? "(无)" : string.Join(",", _preExisting)));

                string winword = FindWinWordExe();
                Log("WINWORD 路径: " + (winword ?? "(未找到)"));
                if (winword == null) return FailAndCleanup("未找到 WINWORD.EXE，请确认已安装 Microsoft Word。");

                string bitnessError;
                if (!WinWordBitnessMatches(winword, out bitnessError))
                    return FailAndCleanup(bitnessError);

                // 生成一个最小空白文档，让 Word 打开真实文档窗口（而非“开始页”）
                string blank = CreateBlankDocx();
                _blankDocxPath = blank;
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
            catch (OperationCanceledException)
            {
                LastError = CancelledSentinel;
                Log("StartAsync: 已取消，不弹出嵌入失败");
                throw;
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
                    if (_stopping) return;   // FormClosing/Stop 已在还原，勿二次 Quit
                    Log("Word 进程已退出 PID=" + pid);
                    _embedded = false;
                    try
                    {
                        if (_resizeDebounce != null)
                        {
                            _resizeDebounce.Stop();
                            _resizeDebounce.Enabled = false;
                        }
                    }
                    catch { }
                    // 最佳努力：仅在正在退出的实例上还原剥壳，不另启 WINWORD
                    if (_chrome.Captured)
                        TryRestoreChromeOnUi(_hwnd);
                    try { CleanupTempDocuments(); } catch { }
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
            // 与 StartAsync 相同契约：先停旧进程并等待，再启动新实例
            EnsureHostedWordStopped(true);
            return await StartAsync();
        }

        /// <summary>
        /// 菜单粘贴：按已锁定 HWND 取该 Word 进程的对象模型，调用 Selection.Paste()。
        /// 与隔离前相同的 COM 粘贴路径，但不走全局 Word.Application。
        /// </summary>
        public void Paste()
        {
            // Word COM 只能在 UI/STA 上调用
            RunOnUi(new Action(PasteCore));
        }

        private void PasteCore()
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

        /// <summary>只结束我们自己的 Word 进程。默认在 UI/STA 还原界面后 Quit，短超时再 Kill。</summary>
        public void Stop()
        {
            Stop(forceKill: false);
        }

        /// <summary>
        /// 结束自有 Word 进程。
        /// forceKill=true：可选先还原已保存的界面属性，再 Kill 并等待退出
        /// （失败启动 / Dispose 兜底，不走 Application.Quit）。
        /// forceKill=false：在 UI/STA 上还原剥壳属性、Close、Quit，短超时后 Kill。
        /// 对象模型绝不放到线程池；超时 Kill 可以在调用线程等待。
        /// </summary>
        public void Stop(bool forceKill)
        {
            // 一开始就禁止 Relayout/剥壳，避免优雅退出等待期间二次 Strip 把还原冲掉
            _stopping = true;
            _embedded = false;
            try
            {
                if (_resizeDebounce != null)
                {
                    _resizeDebounce.Stop();
                    _resizeDebounce.Enabled = false;
                }
            }
            catch (Exception ex) { Log("Stop: 停止 resize 定时器(忽略): " + ex.Message); }

            int pid = _pid;
            IntPtr hwnd = _hwnd;
            var cts = _cts;
            // Cancel 容错：Dispose 流程中 _cts 可能已被释放，直接 Cancel 会抛 ObjectDisposedException
            try { if (cts != null) cts.Cancel(); }
            catch (Exception ex) { Log("Stop: Cancel 异常(忽略): " + ex.Message); }

            if (pid != 0)
            {
                // 先解绑退出事件，避免 Stop 后旧进程 Exited 再触发状态变化
                try
                {
                    if (_wordProcess != null && _exitedHandler != null)
                        _wordProcess.Exited -= _exitedHandler;
                    if (_wordProcess != null)
                        _wordProcess.EnableRaisingEvents = false;
                }
                catch (Exception ex) { Log("解绑 Exited 事件(忽略): " + ex.Message); }

                if (forceKill)
                {
                    // 失败路径不 fire-and-forget Quit；若已剥壳则尽量还原再 Kill
                    TryRestoreChromeOnUi(hwnd);
                    KillProcess(pid);
                    WaitForProcessExit(pid, FORCE_KILL_WAIT_MS);
                }
                else
                {
                    TryCloseGracefully(pid, hwnd);
                }
            }

            // 释放 Process 句柄（不影响进程本体；优雅退出用的是已捕获的 pid 局部变量）
            try { if (_wordProcess != null) _wordProcess.Dispose(); } catch { }

            _pid = 0;
            _wordProcess = null;
            _hwnd = IntPtr.Zero;
            _embedded = false;
            _cachedWwF = _cachedWwB = _cachedWwG = IntPtr.Zero;
            _chrome.Captured = false;

            // 进程已退出后再删空白文档与 Word 锁文件
            CleanupTempDocuments();
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
                        try { proc.WaitForExit(FORCE_KILL_WAIT_MS); } catch { }
                    }
                }
            }
            catch (Exception ex) { Log("kill 异常(忽略): " + ex.Message); }
        }

        /// <summary>等待指定 PID 退出；超时返回 false。不触及 Word OM。</summary>
        private static bool WaitForProcessExit(int pid, int timeoutMs)
        {
            if (pid == 0) return true;
            try
            {
                using (var proc = Process.GetProcessById(pid))
                    return proc.WaitForExit(timeoutMs);
            }
            catch
            {
                return !ProcessAlive(pid);
            }
        }

        /// <summary>
        /// 等待进程退出，不泵 WinForms 消息（避免 OnResize/Timer 在 Quit 期间再次剥壳）。
        /// Restore+Quit 已在 UI/STA 发出；此处只等进程句柄。
        /// </summary>
        private bool WaitForProcessExitPump(int pid, int timeoutMs)
        {
            return WaitForProcessExit(pid, timeoutMs);
        }

        /// <summary>本宿主锁定的 Word 进程是否仍在运行。</summary>
        private bool IsHostProcessAlive()
        {
            try
            {
                if (_wordProcess != null && !_wordProcess.HasExited)
                    return true;
            }
            catch { }
            return _pid != 0 && ProcessAlive(_pid);
        }

        /// <summary>停掉旧宿主并等待退出（启动前 / 新建文档：先停旧再启新）。</summary>
        private void EnsureHostedWordStopped(bool forceKill)
        {
            if (_pid == 0 && !IsHostProcessAlive())
                return;
            Stop(forceKill);
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
        /// 在 UI/STA 上还原界面、Close、Quit；再在调用线程 WaitForExit，超时 Kill。
        /// 不得 Task.Run 到线程池去调 OM。
        /// </summary>
        private void TryCloseGracefully(int pid, IntPtr hwnd)
        {
            if (pid == 0) return;

            RunOnUi(new Action(delegate
            {
                RestoreCloseAndQuit(hwnd);
            }));

            if (!WaitForProcessExitPump(pid, GRACEFUL_QUIT_TIMEOUT_MS))
            {
                Log("graceful: " + (GRACEFUL_QUIT_TIMEOUT_MS / 1000) + "s 内未退出，回退 Kill");
                KillProcess(pid);
                WaitForProcessExitPump(pid, FORCE_KILL_WAIT_MS);
            }
            else
            {
                Log("退出方式=graceful (PID=" + pid + ")");
            }
        }

        /// <summary>必须在 UI/STA：还原剥壳属性、关闭文档、Application.Quit。</summary>
        private void RestoreCloseAndQuit(IntPtr hwnd)
        {
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
                        RestoreChrome(app);

                        object doc = null;
                        try
                        {
                            doc = ComGet(app, "ActiveDocument");
                            if (doc != null)
                            {
                                // wdDoNotSaveChanges：不弹“是否保存”，也不丢给下次恢复
                                doc.GetType().InvokeMember("Close",
                                    BindingFlags.InvokeMethod | BindingFlags.Instance,
                                    null, doc, new object[] { wdDoNotSaveChanges });
                                Log("graceful: ActiveDocument.Close(不保存) OK");
                            }
                        }
                        catch (Exception ex) { Log("graceful: ActiveDocument.Close: " + ex.Message); }
                        finally { if (doc != null) ReleaseCom(doc); }   // Close 抛异常也必须释放 RCW，避免泄漏

                        app.GetType().InvokeMember("Quit",
                            BindingFlags.InvokeMethod | BindingFlags.Instance, null, app, null);
                        Log("graceful: Application.Quit 已发送");
                    }
                    finally { ReleaseCom(app); }
                }
                else
                {
                    Log("graceful: 未绑定到对象模型，跳过 Quit（随后 Kill）");
                }
            }
            catch (Exception ex) { Log("graceful: 退出异常(转 Kill): " + ex.GetType().Name + ": " + ex.Message); }
            finally { ReleaseCom(om); }
        }

        /// <summary>失败/强制路径：若已剥壳，尽量在 UI 上还原，不调用 Quit。</summary>
        private void TryRestoreChromeOnUi(IntPtr hwnd)
        {
            if (!_chrome.Captured) return;
            try
            {
                RunOnUi(new Action(delegate
                {
                    object om = null;
                    try
                    {
                        om = TryBindOmForRestore(hwnd);
                        if (om == null)
                        {
                            Log("TryRestoreChromeOnUi: 句柄已失效，无法在退出实例上还原剥壳属性");
                            return;
                        }
                        object app = ComGet(om, "Application");
                        try { RestoreChrome(app); }
                        finally { ReleaseCom(app); }
                    }
                    catch (Exception ex) { Log("TryRestoreChromeOnUi: " + ex.Message); }
                    finally { ReleaseCom(om); }
                }));
            }
            catch (Exception ex) { Log("TryRestoreChromeOnUi invoke: " + ex.Message); }
        }

        /// <summary>最佳努力绑定正在退出实例的 NativeOM；不另启 WINWORD。</summary>
        private object TryBindOmForRestore(IntPtr hwnd)
        {
            if (hwnd != IntPtr.Zero && NativeMethods.IsWindow(hwnd))
            {
                object om = BindWordNativeOm(hwnd);
                if (om != null) return om;
            }
            if (_cachedWwG != IntPtr.Zero && NativeMethods.IsWindow(_cachedWwG))
            {
                object om = BindWordNativeOm(_cachedWwG);
                if (om != null) return om;
            }
            if (_hwnd != IntPtr.Zero && _hwnd != hwnd && NativeMethods.IsWindow(_hwnd))
            {
                object om = BindWordNativeOm(_hwnd);
                if (om != null) return om;
            }
            return null;
        }

        // ==================== 内部实现 ====================

        /// <summary>启动前快照当前所有 WINWORD 进程的 PID。</summary>
        private void SnapshotPreExisting()
        {
            _preExisting.Clear();
            Process[] procs = null;
            try
            {
                procs = Process.GetProcessesByName("WINWORD");
                if (procs != null)
                {
                    for (int i = 0; i < procs.Length; i++)
                        _preExisting.Add(procs[i].Id);
                }
            }
            catch { }
            finally
            {
                if (procs != null)
                {
                    for (int i = 0; i < procs.Length; i++)
                    {
                        try { procs[i].Dispose(); } catch { }
                    }
                }
            }
        }

        /// <summary>定位 WINWORD.EXE 的完整路径。</summary>
        private static string FindWinWordExe()
        {
            // 先从注册表探查（机器安装 + 当前用户安装 + Click-to-Run）
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
            {
                string trusted = TryValidateWinWordPath(c, "fallback");
                if (trusted != null) return trusted;
            }

            // 兜底：从标准安装路径探查
            try
            {
                string office = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86);
                string p = Path.Combine(office, @"Microsoft Office\root\Office16\WINWORD.EXE");
                string trusted = TryValidateWinWordPath(p, "ProgramFilesX86");
                if (trusted != null) return trusted;
                Log("FindWinWordExe: 兜底标准路径不存在或不信任: " + p);
            }
            catch (Exception ex)
            {
                Log("FindWinWordExe: 构造/检查兜底路径异常: " +
                    ex.GetType().Name + ": " + ex.Message);
            }
            return null;
        }

        /// <summary>
        /// 从 HKLM / HKCU 的 App Paths 以及 Click-to-Run 安装根读取 WINWORD.EXE。
        /// 命中路径必须通过 TryValidateWinWordPath，防止 App Paths 劫持。
        /// </summary>
        private static string FindWinWordFromRegistry()
        {
            string[] appPathKeys =
            {
                @"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\winword.exe",
                @"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths\winword.exe",
            };
            string[] clickKeys =
            {
                @"SOFTWARE\Microsoft\Office\ClickToRun\Configuration",
                @"SOFTWARE\WOW6432Node\Microsoft\Office\ClickToRun\Configuration",
            };

            RegistryKey[] hives = new RegistryKey[] { Registry.LocalMachine, Registry.CurrentUser };
            string[] hiveNames = new string[] { "HKLM", "HKCU" };

            for (int h = 0; h < hives.Length; h++)
            {
                for (int i = 0; i < appPathKeys.Length; i++)
                {
                    string found = TryReadAppPaths(hives[h], hiveNames[h], appPathKeys[i]);
                    if (found != null) return found;
                }
            }
            for (int h = 0; h < hives.Length; h++)
            {
                for (int i = 0; i < clickKeys.Length; i++)
                {
                    string found = TryReadClickToRun(hives[h], hiveNames[h], clickKeys[i]);
                    if (found != null) return found;
                }
            }
            return null;
        }

        private static string TryReadAppPaths(RegistryKey hive, string hiveName, string sub)
        {
            try
            {
                using (RegistryKey key = hive.OpenSubKey(sub))
                {
                    if (key == null)
                    {
                        Log("FindWinWordFromRegistry: 注册表子项不存在: " + hiveName + "\\" + sub);
                        return null;
                    }
                    string v = key.GetValue(null) as string;
                    return TryValidateWinWordPath(v, hiveName + " App Paths");
                }
            }
            catch (Exception ex)
            {
                Log("FindWinWordFromRegistry: 读取注册表异常(" + hiveName + "\\" + sub + "): " +
                    ex.GetType().Name + ": " + ex.Message);
                return null;
            }
        }

        private static string TryReadClickToRun(RegistryKey hive, string hiveName, string sub)
        {
            try
            {
                using (RegistryKey key = hive.OpenSubKey(sub))
                {
                    if (key == null) return null;
                    string install = key.GetValue("InstallationPath") as string;
                    if (string.IsNullOrEmpty(install))
                        install = key.GetValue("InstallPath") as string;
                    string client = key.GetValue("ClientFolder") as string;

                    string[] rels = new string[]
                    {
                        @"root\Office16\WINWORD.EXE",
                        @"Office16\WINWORD.EXE",
                        @"root\Office15\WINWORD.EXE",
                        @"Office15\WINWORD.EXE",
                    };
                    if (!string.IsNullOrEmpty(install))
                    {
                        for (int i = 0; i < rels.Length; i++)
                        {
                            string p = Path.Combine(install, rels[i]);
                            string trusted = TryValidateWinWordPath(p, hiveName + " ClickToRun Install");
                            if (trusted != null) return trusted;
                        }
                    }
                    if (!string.IsNullOrEmpty(client))
                    {
                        string p = Path.Combine(client, "WINWORD.EXE");
                        string trusted = TryValidateWinWordPath(p, hiveName + " ClickToRun ClientFolder");
                        if (trusted != null) return trusted;
                    }
                }
            }
            catch (Exception ex)
            {
                Log("FindWinWordFromRegistry: ClickToRun 异常(" + hiveName + "\\" + sub + "): " +
                    ex.GetType().Name + ": " + ex.Message);
            }
            return null;
        }

        /// <summary>
        /// 拒绝 App Paths 劫持：必须存在、文件名是 WINWORD.EXE、路径落在 Office 目录下。
        /// </summary>
        private static string TryValidateWinWordPath(string raw, string source)
        {
            if (string.IsNullOrEmpty(raw)) return null;
            try
            {
                string trimmed = raw.Trim().Trim('"');
                if (trimmed.Length == 0) return null;
                string full = Path.GetFullPath(trimmed);
                if (!File.Exists(full))
                {
                    Log("FindWinWordExe: 路径不存在 (" + source + "): " + full);
                    return null;
                }
                string name = Path.GetFileName(full);
                if (!string.Equals(name, "WINWORD.EXE", StringComparison.OrdinalIgnoreCase))
                {
                    Log("FindWinWordExe: 拒绝非 WINWORD.EXE (" + source + "): " + full);
                    return null;
                }
                string lower = full.Replace('/', '\\').ToLowerInvariant();
                bool officeDir = lower.IndexOf("\\microsoft office\\") >= 0
                    || lower.IndexOf("\\office16\\") >= 0
                    || lower.IndexOf("\\office15\\") >= 0;
                if (!officeDir)
                {
                    Log("FindWinWordExe: 拒绝非 Office 目录 (" + source + "): " + full);
                    return null;
                }
                Log("FindWinWordExe: 采用 (" + source + "): " + full);
                return full;
            }
            catch (Exception ex)
            {
                Log("FindWinWordExe: 校验路径异常 (" + source + "): " +
                    ex.GetType().Name + ": " + ex.Message);
                return null;
            }
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
                // 下一次启动也要清 blank-*.docx 与 ~$*.docx（Word 锁 ~$ank-* 匹配不到 blank-*.docx）
                TryDeleteGlob(dir, "blank-*.docx");
                TryDeleteGlob(dir, "~$*.docx");

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

        private static string GetEmbedTempDir()
        {
            return Path.Combine(Path.GetTempPath(), "WordEmbedDemo");
        }

        private static void TryDeleteFile(string path)
        {
            if (string.IsNullOrEmpty(path)) return;
            try { File.Delete(path); }
            catch { }   // 仍被锁定则忽略
        }

        private static void TryDeleteGlob(string dir, string pattern)
        {
            try
            {
                if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir)) return;
                string[] files = Directory.GetFiles(dir, pattern);
                for (int i = 0; i < files.Length; i++)
                    TryDeleteFile(files[i]);
            }
            catch { }
        }

        /// <summary>Quit/Kill 并等待退出后删除本次空白文档及 Word 锁文件 ~$*.docx。</summary>
        private void CleanupTempDocuments()
        {
            try
            {
                string path = _blankDocxPath;
                if (!string.IsNullOrEmpty(path))
                {
                    TryDeleteFile(path);
                    try
                    {
                        string dir = Path.GetDirectoryName(path);
                        string name = Path.GetFileName(path);
                        // Word 锁文件：~$ + 去掉前两个字符（blank-xxx.docx → ~$ank-xxx.docx）
                        if (!string.IsNullOrEmpty(dir) && !string.IsNullOrEmpty(name) && name.Length >= 2)
                            TryDeleteFile(Path.Combine(dir, "~$" + name.Substring(2)));
                        TryDeleteGlob(dir, "~$*.docx");
                    }
                    catch { }
                }
                else
                {
                    TryDeleteGlob(GetEmbedTempDir(), "~$*.docx");
                }
            }
            catch { }
            _blankDocxPath = null;
        }

        /// <summary>
        /// 轮询等待我们锁定进程的真正主框架窗口（类名 OpusApp）出现。
        /// 不再依赖 Process.MainWindowHandle（它可能拿到启动画面等过渡窗口）。
        /// </summary>
        private async Task<bool> WaitForMainWindowAsync()
        {
            var startTime = DateTime.UtcNow;
            // 循环含 await，期间 _cts 字段可能被 Stop/Dispose 替换或释放；
            // 在同步进入时的安全时点捕获 token 为局部变量，避免读字段抛 ODE
            var token = _cts.Token;

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
            token.ThrowIfCancellationRequested();

            var deadline = DateTime.UtcNow.AddSeconds(30);
            int round = 0;
            while (DateTime.UtcNow < deadline)
            {
                token.ThrowIfCancellationRequested();   // 取消与失败区分：抛 OCE，不当嵌入失败
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
                await Task.Delay(200, token);
            }
            token.ThrowIfCancellationRequested();
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
            // 同 WaitForMainWindowAsync：捕获局部 token，避免 await 后读已释放的字段
            var token = _cts.Token;
            int i = 0;
            while (DateTime.UtcNow < deadline)
            {
                token.ThrowIfCancellationRequested();
                i++;
                IntPtr doc = NativeMethods.FindChildWindowRecursive(_hwnd, "_WwG");
                if (doc != IntPtr.Zero)
                {
                    Log("_WwG 文档子窗口已出现 HWND=0x" + doc.ToString("X") + " 等待次数=" + i);
                    return true;
                }
                await Task.Delay(200, token);
            }
            Log("警告: 等待 _WwG 超时(" + timeoutMs + "ms)，继续嵌入流程");
            return false;
        }

        /// <summary>把锁定的主窗口挂靠为本面板子窗口，并铺满、剥壳。</summary>
        private async Task<bool> EmbedWindowAsync()
        {
            IntPtr h = _hwnd;
            if (h == IntPtr.Zero) return FailAndCleanup("无效的 Word 窗口句柄。");
            // 捕获局部 token：下方 await 之后不再读 _cts 字段（防释放竞态）
            var token = _cts.Token;

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
            token.ThrowIfCancellationRequested();   // 等待期间被 Stop，不当嵌入失败
            if (_stopping)
                throw new OperationCanceledException(token);

            // 先置 _embedded，使 Relayout 的守卫能放过初始铺满；Stop 会立刻清回 false
            _embedded = true;
            RelayoutEmbeddedWord();
            if (_stopping)
            {
                _embedded = false;
                throw new OperationCanceledException(token);
            }

            // 铺满后显示
            NativeMethods.ShowWindow(h, NativeMethods.SW_SHOW);
            Log("嵌入完成 hwnd=0x" + h.ToString("X") + " size=" + ClientSize.Width + "x" + ClientSize.Height);
            return true;
        }

        /// <summary>
        /// 嵌入后统一重排：清零 chrome 占位，按 OpusApp → _WwF → _WwB → _WwG 铺满面板，
        /// 再用对象模型关 Ribbon/标尺/任务窗格/状态栏并把纸张 PageFit 居中。
        /// Embed 与 Resize 共用，避免拖拽后又偏回去。
        /// Word 在 resize 后会重新布局内部 UI 并恢复菜单栏/工具栏，因此每次 resize
        /// 都必须重新执行 OM 剥壳，不能只做一次缓存。
        /// </summary>
        private void RelayoutEmbeddedWord()
        {
            if (_stopping || !_embedded) return;
            if (_hwnd == IntPtr.Zero || ClientSize.Width <= 0 || ClientSize.Height <= 0) return;

            NativeMethods.SendMessage(_hwnd, NativeMethods.WM_SETREDRAW, (IntPtr)0, IntPtr.Zero);
            try
            {
                StripWordChrome();
                FillDocumentChain();
                // 每次 resize 都重新执行 OM 剥壳，防止 Word 重新布局时恢复 Ribbon/标尺/状态栏
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
        /// 返回 false 表示 OM 未绑定（Word 未就绪），调用方下次可重试。
        /// </summary>
        private bool StripChromeViaOm()
        {
            if (_stopping || !_embedded) return false;
            // OM 必须在创建本控件的 UI/STA 线程上
            if (IsHandleCreated && InvokeRequired)
                return (bool)Invoke(new Func<bool>(StripChromeViaOmCore));
            return StripChromeViaOmCore();
        }

        private bool StripChromeViaOmCore()
        {
            if (_stopping || !_embedded) return false;
            object om = null;
            try
            {
                om = BindWordNativeOm();
                if (om == null)
                {
                    Log("StripChromeViaOm: 未绑定对象模型，跳过（仅依赖几何剥壳）");
                    return false;
                }

                object app = ComGet(om, "Application");
                try
                {
                    if (!_chrome.Captured)
                        CaptureChrome(app);

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

            // OM 绑定成功即视为本轮剥壳完成（子项失败已降级记日志，不重试）；
            // 返回 false 仅发生在 OM 未就绪时，下次 resize 会重试
            return om != null;
        }

        /// <summary>第一次剥壳前保存 Application/窗口级界面属性。</summary>
        private void CaptureChrome(object app)
        {
            if (app == null || _chrome.Captured) return;
            try
            {
                object cbs = null;
                try
                {
                    cbs = ComGet(app, "CommandBars");
                    _chrome.RibbonEnabled = TryGetCommandBarProp(cbs, "Ribbon", "Enabled");
                    _chrome.TaskPaneVisible = TryGetCommandBarProp(cbs, "Task Pane", "Visible");
                    _chrome.NavigationVisible = TryGetCommandBarProp(cbs, "Navigation", "Visible");
                }
                catch (Exception ex) { Log("CaptureChrome: CommandBars: " + ex.Message); }
                finally { ReleaseCom(cbs); }

                _chrome.DisplayStatusBar = TryComGet(app, "DisplayStatusBar");

                object win = null, view = null, zoom = null;
                try
                {
                    win = ComGet(app, "ActiveWindow");
                    _chrome.DisplayRulers = TryComGet(win, "DisplayRulers");
                    _chrome.DocumentMap = TryComGet(win, "DocumentMap");
                    view = TryComGet(win, "View");
                    if (view != null)
                    {
                        _chrome.ViewType = TryComGet(view, "Type");
                        zoom = TryComGet(view, "Zoom");
                        if (zoom != null)
                            _chrome.PageFit = TryComGet(zoom, "PageFit");
                    }
                }
                catch (Exception ex) { Log("CaptureChrome: ActiveWindow: " + ex.Message); }
                finally
                {
                    ReleaseCom(zoom);
                    ReleaseCom(view);
                    ReleaseCom(win);
                }

                _chrome.Captured = true;
                Log("CaptureChrome: 已保存 Ribbon/StatusBar/Rulers/DocumentMap/View/PageFit");
            }
            catch (Exception ex)
            {
                Log("CaptureChrome: " + ex.GetType().Name + ": " + ex.Message);
            }
        }

        /// <summary>把剥壳改过的属性还原到保存值（必须在 UI/STA 且尚未 Quit）。</summary>
        private void RestoreChrome(object app)
        {
            if (app == null || !_chrome.Captured)
            {
                Log("RestoreChrome: 无已保存快照，跳过");
                return;
            }
            try
            {
                object cbs = null;
                try
                {
                    cbs = ComGet(app, "CommandBars");
                    if (_chrome.RibbonEnabled != null)
                        TrySetCommandBarEnabled(cbs, "Ribbon", Convert.ToBoolean(_chrome.RibbonEnabled));
                    if (_chrome.TaskPaneVisible != null)
                        TrySetCommandBarVisible(cbs, "Task Pane", Convert.ToBoolean(_chrome.TaskPaneVisible));
                    if (_chrome.NavigationVisible != null)
                        TrySetCommandBarVisible(cbs, "Navigation", Convert.ToBoolean(_chrome.NavigationVisible));
                }
                catch (Exception ex) { Log("RestoreChrome: CommandBars: " + ex.Message); }
                finally { ReleaseCom(cbs); }

                if (_chrome.DisplayStatusBar != null)
                {
                    try
                    {
                        ComSet(app, "DisplayStatusBar", _chrome.DisplayStatusBar);
                        Log("RestoreChrome: DisplayStatusBar=" + _chrome.DisplayStatusBar + " OK");
                    }
                    catch (Exception ex) { Log("RestoreChrome: DisplayStatusBar: " + ex.Message); }
                }

                object win = null, view = null, zoom = null;
                try
                {
                    win = ComGet(app, "ActiveWindow");
                    if (_chrome.DisplayRulers != null)
                    {
                        try { ComSet(win, "DisplayRulers", _chrome.DisplayRulers); Log("RestoreChrome: DisplayRulers OK"); }
                        catch (Exception ex) { Log("RestoreChrome: DisplayRulers: " + ex.Message); }
                    }
                    if (_chrome.DocumentMap != null)
                    {
                        try { ComSet(win, "DocumentMap", _chrome.DocumentMap); Log("RestoreChrome: DocumentMap OK"); }
                        catch (Exception ex) { Log("RestoreChrome: DocumentMap: " + ex.Message); }
                    }
                    view = TryComGet(win, "View");
                    if (view != null)
                    {
                        if (_chrome.ViewType != null)
                        {
                            try { ComSet(view, "Type", _chrome.ViewType); Log("RestoreChrome: View.Type OK"); }
                            catch (Exception ex) { Log("RestoreChrome: View.Type: " + ex.Message); }
                        }
                        zoom = TryComGet(view, "Zoom");
                        if (zoom != null && _chrome.PageFit != null)
                        {
                            try { ComSet(zoom, "PageFit", _chrome.PageFit); Log("RestoreChrome: Zoom.PageFit OK"); }
                            catch (Exception ex) { Log("RestoreChrome: Zoom.PageFit: " + ex.Message); }
                        }
                    }
                }
                catch (Exception ex) { Log("RestoreChrome: ActiveWindow: " + ex.Message); }
                finally
                {
                    ReleaseCom(zoom);
                    ReleaseCom(view);
                    ReleaseCom(win);
                }
                Log("RestoreChrome: 完成");
            }
            catch (Exception ex)
            {
                Log("RestoreChrome: " + ex.GetType().Name + ": " + ex.Message);
            }
        }

        private static object TryGetCommandBarProp(object commandBars, string name, string prop)
        {
            object bar = null;
            try
            {
                bar = commandBars.GetType().InvokeMember("Item",
                    BindingFlags.GetProperty | BindingFlags.Instance,
                    null, commandBars, new object[] { name });
                return ComGet(bar, prop);
            }
            catch (Exception ex)
            {
                Log("CaptureChrome: CommandBars(" + name + ")." + prop + ": " + ex.Message);
                return null;
            }
            finally { ReleaseCom(bar); }
        }

        private static object TryComGet(object target, string name)
        {
            if (target == null) return null;
            try { return ComGet(target, name); }
            catch { return null; }
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
            if (_disposed || _stopping) return;
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
                // 先结束自有进程（Stop 内部 Cancel 已容错），再释放 _cts；
                // 原顺序“先 Dispose 再 Stop→Cancel”会在每次关窗时抛 ObjectDisposedException
                Stop(forceKill: true);
                try { if (_cts != null) _cts.Cancel(); } catch { }
                try { if (_cts != null) _cts.Dispose(); } catch { }
                _cts = null;
                if (_resizeDebounce != null)
                {
                    _resizeDebounce.Stop();
                    _resizeDebounce.Dispose();
                }
            }
            base.Dispose(disposing);
        }

        // ==================== 日志/错误 ====================

        private bool Fail(string msg)
        {
            LastError = msg;
            Log("FAIL: " + msg);
            RaiseError(msg);   // 交给订阅者（主窗体）显示，控件不直接弹框
            RaiseState(HostStatus.Failed);
            return false;
        }

        /// <summary>失败出口：强制结束并等待宿主 Word，不 fire-and-forget 优雅 Quit。</summary>
        private bool FailAndCleanup(string msg)
        {
            Stop(forceKill: true);
            return Fail(msg);
        }

        /// <summary>
        /// 把委托封送到创建本控件的 UI/STA 线程。已在 UI 上则直接执行。
        /// Word 对象模型（绑定 / 剥壳 / 还原 / Close / Quit / Paste）必须走这里。
        /// </summary>
        private void RunOnUi(Action action)
        {
            if (action == null) return;
            try
            {
                if (!IsDisposed && IsHandleCreated && InvokeRequired)
                    Invoke(action);
                else
                    action();
            }
            catch (ObjectDisposedException)
            {
                Log("RunOnUi: 控件已释放，跳过 OM");
            }
            catch (InvalidOperationException ex)
            {
                Log("RunOnUi: " + ex.Message);
            }
        }

        /// <summary>
        /// 读取 WINWORD.EXE 的 PE COFF Machine，与本进程位数比对。
        /// 0x14C = x86，0x8664 = x64。不一致则拒绝半嵌入。
        /// </summary>
        private static bool WinWordBitnessMatches(string winword, out string error)
        {
            error = null;
            ushort machine;
            try
            {
                machine = ReadPeMachine(winword);
            }
            catch (Exception ex)
            {
                error = "无法读取 WINWORD.EXE 的 PE 头: " + ex.Message;
                return false;
            }

            bool word64 = machine == IMAGE_FILE_MACHINE_AMD64;
            bool word32 = machine == IMAGE_FILE_MACHINE_I386;
            bool proc64 = Environment.Is64BitProcess;
            Log("Bitness: WINWORD Machine=0x" + machine.ToString("X") +
                " word64=" + word64 + " process64=" + proc64 + " IntPtr.Size=" + IntPtr.Size);

            if (!word64 && !word32)
            {
                error = "无法识别 WINWORD.EXE 的体系结构（COFF Machine=0x" + machine.ToString("X") + "）。";
                return false;
            }
            if (word64 != proc64)
            {
                error = "本程序是 " + (proc64 ? "64" : "32") + " 位进程，但 WINWORD.EXE 是 " +
                        (word64 ? "64" : "32") + " 位。位数不一致无法嵌入，请使用 x64 生成并匹配已安装的 Office。";
                return false;
            }
            return true;
        }

        private static ushort ReadPeMachine(string exePath)
        {
            using (var fs = new FileStream(exePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            using (var br = new BinaryReader(fs))
            {
                if (br.ReadUInt16() != 0x5A4D)   // MZ
                    throw new InvalidDataException("不是有效的 PE（缺少 MZ）");
                fs.Seek(0x3C, SeekOrigin.Begin);
                int lfanew = br.ReadInt32();
                fs.Seek(lfanew, SeekOrigin.Begin);
                uint sig = br.ReadUInt32();
                if (sig != 0x00004550)           // PE\0\0
                    throw new InvalidDataException("不是有效的 PE（缺少 PE 签名）");
                return br.ReadUInt16();          // COFF Machine
            }
        }

        /// <summary>上抛错误给宿主窗体（约定在 UI 线程调用）。窗体已销毁时静默丢弃，避免幽灵弹框。</summary>
        private void RaiseError(string msg)
        {
            if (_disposed) return;
            if (!string.IsNullOrEmpty(msg))
                LastError = msg;
            var eh = HostError;
            if (eh != null) eh(msg);
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
