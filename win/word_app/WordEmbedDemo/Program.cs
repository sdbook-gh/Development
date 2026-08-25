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
            // 声明 Per-Monitor V2 DPI 感知（失败回退系统级），保证宿主与 Word 坐标体系一致
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
    }
}
