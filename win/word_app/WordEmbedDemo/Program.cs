using System;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace WordEmbedDemo
{
    internal static class Program
    {
        [DllImport("user32.dll")]
        private static extern bool SetProcessDpiAwarenessContext(IntPtr value);

        [DllImport("user32.dll")]
        private static extern bool SetProcessDPIAware();

        [STAThread]
        static void Main()
        {
            // 全局异常兜底：嵌入流程跳进程 P/Invoke + 反射，任何漏网异常都写入同一份日志，
            // 避免“双击没反应 / 直接闪退”无法定位
            Application.ThreadException += (s, e) => Report("UI", e.Exception);
            AppDomain.CurrentDomain.UnhandledException += (s, e) => Report("AppDomain", e.ExceptionObject as Exception);
#if !DEBUG
            // Release 下让 WinForms 吃掉 UI 线程异常（走上面的 ThreadException），而不是弹 JIT 调试框
            try { Application.SetUnhandledExceptionMode(UnhandledExceptionMode.CatchException); } catch { }
#endif

            // 声明 Per-Monitor V2 DPI 感知（失败回退系统级），保证宿主与 Word 坐标体系一致。
            // 注：app.manifest 已声明 PerMonitorV2，进程启动时即生效；下面的调用仅作双保险，
            // manifest 已生效时返回 false 属正常现象，不能当作错误处理。
            try
            {
                if (!SetProcessDpiAwarenessContext((IntPtr)(-4)))
                    SetProcessDPIAware();
            }
            catch
            {
                try { SetProcessDPIAware(); } catch { }
            }

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm());
        }

        /// <summary>未处理异常统一入口：写日志（与嵌入日志同一文件，保证单一时间线）+ 提示。</summary>
        private static void Report(string source, Exception ex)
        {
            try
            {
                WordProcessHost.Log("UNHANDLED[" + source + "]: " +
                    (ex == null ? "(null)" : ex.ToString()));
            }
            catch { }

            try
            {
                MessageBox.Show("发生未处理异常（详见 word_embed_log.txt）：\n"
                        + (ex == null ? "未知错误" : ex.Message),
                    "错误", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            catch { }
        }
    }
}
