using System;
using System.Runtime.InteropServices;
using System.Text;

namespace WordEmbedDemo
{
    /// <summary>
    /// Win32 API 声明，用于把外部进程（Word）的窗口设置为当前窗体的子窗口。
    /// </summary>
    internal static class NativeMethods
    {
        [DllImport("user32.dll", SetLastError = true)]
        public static extern IntPtr SetParent(IntPtr hWndChild, IntPtr hWndNewParent);

        [DllImport("user32.dll", SetLastError = true)]
        public static extern bool MoveWindow(IntPtr hWnd, int X, int Y, int nWidth, int nHeight, bool bRepaint);

        [DllImport("user32.dll", EntryPoint = "GetWindowLong", SetLastError = true)]
        private static extern int GetWindowLong32(IntPtr hWnd, int nIndex);

        [DllImport("user32.dll", EntryPoint = "SetWindowLong", SetLastError = true)]
        private static extern int SetWindowLong32(IntPtr hWnd, int nIndex, int dwNewLong);

        [DllImport("user32.dll", EntryPoint = "GetWindowLongPtr", SetLastError = true)]
        private static extern IntPtr GetWindowLong64(IntPtr hWnd, int nIndex);

        [DllImport("user32.dll", EntryPoint = "SetWindowLongPtr", SetLastError = true)]
        private static extern IntPtr SetWindowLong64(IntPtr hWnd, int nIndex, IntPtr dwNewLong);

        [DllImport("user32.dll")]
        public static extern bool EnumWindows(EnumChildProc lpEnumFunc, IntPtr lParam);

        [DllImport("user32.dll")]
        public static extern bool EnumChildWindows(IntPtr hWndParent, EnumChildProc lpEnumFunc, IntPtr lParam);

        public delegate bool EnumChildProc(IntPtr hWnd, IntPtr lParam);

        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        public static extern int GetClassName(IntPtr hWnd, [Out] StringBuilder lpClassName, int nMaxCount);

        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        public static extern IntPtr FindWindowEx(IntPtr hWndParent, IntPtr hWndChildAfter,
            string lpszClass, string lpszWindow);

        /// <summary>获取窗口类名，失败返回空字符串。</summary>
        public static string GetWindowClassName(IntPtr hWnd)
        {
            var sb = new StringBuilder(256);
            return GetClassName(hWnd, sb, sb.Capacity) > 0 ? sb.ToString() : string.Empty;
        }

        /// <summary>
        /// 递归在 parent 的所有后代窗口中按类名查找，优先返回可见者。
        /// 注意：FindWindowEx 只查【直接子窗口】，Word 的 OpusApp &gt; _WwF &gt; _WwB &gt; _WwG
        /// 必须靠 EnumChildWindows（递归）才能拿到 _WwG。
        /// </summary>
        public static IntPtr FindChildWindowRecursive(IntPtr parent, string className)
        {
            IntPtr found = IntPtr.Zero;
            IntPtr visible = IntPtr.Zero;
            if (parent == IntPtr.Zero || string.IsNullOrEmpty(className)) return IntPtr.Zero;
            EnumChildWindows(parent, (h, l) =>
            {
                if (GetWindowClassName(h) != className) return true;
                if (found == IntPtr.Zero) found = h;
                if (IsWindowVisible(h))
                {
                    visible = h;
                    return false;
                }
                return true;
            }, IntPtr.Zero);
            return visible != IntPtr.Zero ? visible : found;
        }

        [DllImport("user32.dll")]
        public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

        [DllImport("user32.dll", SetLastError = true)]
        public static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter,
            int X, int Y, int cx, int cy, uint uFlags);

        [DllImport("user32.dll", SetLastError = true)]
        public static extern IntPtr SendMessage(IntPtr hWnd, uint Msg, IntPtr wParam, IntPtr lParam);

        [DllImport("user32.dll")]
        public static extern bool InvalidateRect(IntPtr hWnd, IntPtr lpRect, bool bErase);

        // 以下两项当前流程未直接调用，预留给“恢复/还原嵌入窗口”等后续实机调试
        [DllImport("user32.dll")]
        public static extern bool SetForegroundWindow(IntPtr hWnd);

        [DllImport("user32.dll")]
        public static extern bool IsWindow(IntPtr hWnd);

        [DllImport("user32.dll")]
        public static extern bool IsWindowVisible(IntPtr hWnd);

        [DllImport("user32.dll")]
        public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);

        /// <summary>
        /// 从指定 HWND 取 Office 原生对象模型（IDispatch）。
        /// 对 Word 的 _WwG / OpusApp 传入 OBJID_NATIVEOM，得到该窗口所属的 Window/Application，
        /// 而不会连到 ROT 里的全局 Word.Application。
        /// </summary>
        [DllImport("oleacc.dll")]
        public static extern int AccessibleObjectFromWindow(IntPtr hwnd, uint dwObjectID,
            ref Guid riid, [MarshalAs(UnmanagedType.IUnknown)] out object ppvObject);

        public static readonly Guid IID_IDispatch = new Guid("00020400-0000-0000-C000-000000000046");
        public const uint OBJID_NATIVEOM = 0xFFFFFFF0;

        public const int SW_HIDE = 0;
        public const int SW_SHOWNORMAL = 1;   // 预留：还原显示嵌入窗口时使用
        public const int SW_SHOW = 5;
        public const int SW_RESTORE = 9;      // 预留：最小化恢复时使用

        public const uint SWP_NOSIZE = 0x0001;
        public const uint SWP_NOMOVE = 0x0002;
        public const uint SWP_NOZORDER = 0x0004;
        public const uint SWP_FRAMECHANGED = 0x0020;

        public const uint WM_SETREDRAW = 0x000B;

        public const int GWL_STYLE = -16;

        // 窗口样式位
        public const int WS_CHILD = 0x40000000;
        public const int WS_VISIBLE = 0x10000000;
        public const int WS_CAPTION = 0x00C00000;
        public const int WS_THICKFRAME = 0x00040000;
        public const int WS_SYSMENU = 0x00080000;
        public const int WS_MINIMIZEBOX = 0x00020000;
        public const int WS_MAXIMIZEBOX = 0x00010000;

        public static int GetWindowStyle(IntPtr hWnd)
        {
            if (IntPtr.Size == 8)
                return unchecked((int)(long)GetWindowLong64(hWnd, GWL_STYLE));
            return GetWindowLong32(hWnd, GWL_STYLE);
        }

        public static IntPtr SetWindowStyle(IntPtr hWnd, int style)
        {
            if (IntPtr.Size == 8)
                return SetWindowLong64(hWnd, GWL_STYLE, (IntPtr)style);
            return (IntPtr)SetWindowLong32(hWnd, GWL_STYLE, style);
        }
    }
}