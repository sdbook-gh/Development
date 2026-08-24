using System;
using System.Windows.Forms;

namespace WordEmbedDemo
{
    /// <summary>
    /// 承载 Word 窗口的容器控件：负责在自身大小变化时通知宿主调整 Word 子窗口的位置与尺寸。
    /// </summary>
    public class WordHostControl : Panel
    {
        private readonly Action<IntPtr> _onResizeHostedWindow;

        public WordHostControl(Action<IntPtr> onResizeHostedWindow)
        {
            if (onResizeHostedWindow == null)
                throw new ArgumentNullException("onResizeHostedWindow");
            _onResizeHostedWindow = onResizeHostedWindow;
            Dock = DockStyle.Fill;
            BackColor = System.Drawing.Color.White;
        }

        protected override void OnResize(EventArgs eventargs)
        {
            base.OnResize(eventargs);
            _onResizeHostedWindow(Handle);
        }
    }
}
