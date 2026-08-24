using System;
using System.Runtime.InteropServices;

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

        public static void SetWindowStyle(IntPtr hWnd, int style)
        {
            if (IntPtr.Size == 8)
                SetWindowLong64(hWnd, GWL_STYLE, (IntPtr)style);
            else
                SetWindowLong32(hWnd, GWL_STYLE, style);
        }
    }
}
