using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace WordEmbedDemo
{
    /// <summary>
    /// 主窗体：通过 COM 晚期绑定启动本机安装的 Microsoft Word，
    /// 并使用 SetParent 将其窗口嵌入到本窗体的面板中，支持粘贴等编辑功能。
    /// </summary>
    public partial class MainForm : Form
    {
        private dynamic _wordApp;          // Word.Application (COM 晚期绑定)
        private object _missing = Type.Missing;
        private IntPtr _wordHwnd = IntPtr.Zero;
        private bool _embedded;
        private string _currentFile;

        private MenuStrip _menu;
        private ToolStripMenuItem _menuFile;
        private ToolStripMenuItem _miOpen;
        private ToolStripMenuItem _miNew;
        private ToolStripMenuItem _miSave;
        private ToolStripMenuItem _menuEdit;
        private ToolStripMenuItem _miPaste;
        private WordHostControl _hostPanel;
        private StatusStrip _status;
        private ToolStripStatusLabel _lblStatus;

        public MainForm()
        {
            Text = "Word 嵌入示例 (.NET Framework 4.8)";
            Width = 1000;
            Height = 720;
            StartPosition = FormStartPosition.CenterScreen;
            WindowState = FormWindowState.Maximized;

            BuildUi();

            Load += MainForm_Load;
            FormClosing += MainForm_FormClosing;
        }

        private void BuildUi()
        {
            // 菜单栏：文件 / 编辑
            _menu = new MenuStrip();
            _menuFile = new ToolStripMenuItem("文件(&F)");
            _miNew = new ToolStripMenuItem("新建文档(&N)", null, (s, e) => NewDocument());
            _miOpen = new ToolStripMenuItem("打开文档(&O)...", null, (s, e) => OpenDocument());
            _miSave = new ToolStripMenuItem("保存文档(&S)...", null, (s, e) => SaveDocument());
            _menuFile.DropDownItems.Add(_miNew);
            _menuFile.DropDownItems.Add(_miOpen);
            _menuFile.DropDownItems.Add(_miSave);

            _menuEdit = new ToolStripMenuItem("编辑(&E)");
            _miPaste = new ToolStripMenuItem("粘贴(&P)  Ctrl+V", null, (s, e) => PasteFromClipboard());
            _menuEdit.DropDownItems.Add(_miPaste);

            _menu.Items.Add(_menuFile);
            _menu.Items.Add(_menuEdit);

            // 状态栏
            _status = new StatusStrip();
            _lblStatus = new ToolStripStatusLabel("就绪");
            _status.Items.Add(_lblStatus);

            // Word 宿主面板（停靠在菜单栏下方、状态栏上方）
            _hostPanel = new WordHostControl(OnHostPanelResize);

            Controls.Add(_hostPanel);
            Controls.Add(_status);
            Controls.Add(_menu);
            MainMenuStrip = _menu;
        }

        // ------------------------------------------------------------------
        // 启动并嵌入 Word
        // ------------------------------------------------------------------
        private void MainForm_Load(object sender, EventArgs e)
        {
            try
            {
                var wordType = Type.GetTypeFromProgID("Word.Application");
                if (wordType == null)
                {
                    MessageBox.Show("未检测到系统安装的 Microsoft Word。", "错误",
                        MessageBoxButtons.OK, MessageBoxIcon.Error);
                    Close();
                    return;
                }

                SetStatus("正在启动 Word...");
                _wordApp = Activator.CreateInstance(wordType);
                _wordApp.Visible = true;              // 必须可见，否则拿不到窗口句柄/无法正常渲染
                _wordApp.DisplayAlerts = 0;           // wdAlertsNone

                // 新建一个空文档作为编辑区（COM 可选参数，直接不传即可，无需 ref）
                _wordApp.Documents.Add();

                // 取得 Word 文档窗口的句柄（Word.Window.Hwnd）
                _wordHwnd = (IntPtr)(int)_wordApp.ActiveWindow.Hwnd;
                if (_wordHwnd == IntPtr.Zero)
                {
                    MessageBox.Show("无法获取 Word 窗口句柄。", "错误",
                        MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }

                EmbedWordWindow();
                SetStatus("Word 已嵌入，可直接在区域内编辑；Ctrl+C/Ctrl+V/X 均可用。");
            }
            catch (COMException ex)
            {
                MessageBox.Show("启动 Word 失败：" + ex.Message, "错误",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
                Close();
            }
        }

        private void EmbedWordWindow()
        {
            // 1) 把 Word 窗口设为本窗体面板的子窗口
            NativeMethods.SetParent(_wordHwnd, _hostPanel.Handle);

            // 2) 调整样式：去掉标题栏/边框/系统菜单，加上 WS_CHILD
            int style = NativeMethods.GetWindowStyle(_wordHwnd);
            style &= ~(NativeMethods.WS_CAPTION |
                       NativeMethods.WS_THICKFRAME |
                       NativeMethods.WS_SYSMENU |
                       NativeMethods.WS_MINIMIZEBOX |
                       NativeMethods.WS_MAXIMIZEBOX);
            style |= NativeMethods.WS_CHILD | NativeMethods.WS_VISIBLE;
            NativeMethods.SetWindowStyle(_wordHwnd, style);

            // 3) 铺满宿主面板
            ResizeWordToPanel();
            _embedded = true;
        }

        private void OnHostPanelResize(IntPtr hostHandle)
        {
            ResizeWordToPanel();
        }

        private void ResizeWordToPanel()
        {
            if (_wordHwnd == IntPtr.Zero || !_embedded || _hostPanel.Width <= 0 || _hostPanel.Height <= 0)
                return;
            NativeMethods.MoveWindow(_wordHwnd, 0, 0, _hostPanel.ClientSize.Width,
                                     _hostPanel.ClientSize.Height, true);
        }

        // ------------------------------------------------------------------
        // 编辑功能
        // ------------------------------------------------------------------

        /// <summary>把剪贴板内容粘贴进 Word 光标位置。</summary>
        private void PasteFromClipboard()
        {
            EnsureWordAlive();
            try
            {
                // 方式一：调用 Word 自身的 Paste（保留格式）
                _wordApp.ActiveWindow.Selection.Paste();

                // 若剪贴板是纯文本且不想带格式，可改用：
                // string text = Clipboard.GetText();
                // _wordApp.ActiveWindow.Selection.TypeText(text);
            }
            catch (COMException ex)
            {
                MessageBox.Show("粘贴失败：" + ex.Message, "提示",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        private void NewDocument()
        {
            EnsureWordAlive();
            try { _wordApp.Documents.Add(); }
            catch (COMException ex) { ShowComError(ex); }
        }

        private void OpenDocument()
        {
            EnsureWordAlive();
            using (var dlg = new OpenFileDialog())
            {
                dlg.Filter = "Word 文档 (*.doc;*.docx)|*.doc;*.docx|所有文件 (*.*)|*.*";
                if (dlg.ShowDialog(this) != DialogResult.OK) return;
                try
                {
                    // 仅传文件名，其余可选参数由 COM 调用器自动补 Type.Missing
                    _wordApp.Documents.Open(dlg.FileName);
                    _currentFile = dlg.FileName;
                    RebindWordWindow();   // 打开新文档后活动窗口句柄可能变化，重新绑定
                    SetStatus("已打开：" + Path.GetFileName(dlg.FileName));
                }
                catch (COMException ex) { ShowComError(ex); }
            }
        }

        private void SaveDocument()
        {
            EnsureWordAlive();
            try
            {
                if (!string.IsNullOrEmpty(_currentFile))
                    _wordApp.ActiveDocument.Save();
                else
                {
                    using (var dlg = new SaveFileDialog())
                    {
                        dlg.Filter = "Word 文档 (*.docx)|*.docx";
                        if (dlg.ShowDialog(this) == DialogResult.OK)
                            _wordApp.ActiveDocument.SaveAs2(dlg.FileName);
                    }
                }
                SetStatus("已保存。");
            }
            catch (COMException ex) { ShowComError(ex); }
        }

        /// <summary>切换文档后 ActiveWindow 可能变化，重新获取句柄并重新挂载。</summary>
        private void RebindWordWindow()
        {
            try
            {
                IntPtr hwnd = (IntPtr)(int)_wordApp.ActiveWindow.Hwnd;
                if (hwnd != _wordHwnd)
                {
                    _wordHwnd = hwnd;
                    _embedded = false;
                    EmbedWordWindow();
                }
                else
                {
                    ResizeWordToPanel();
                }
            }
            catch (COMException ex) { ShowComError(ex); }
        }

        // ------------------------------------------------------------------
        // 关闭清理：还原 Word 窗口 → 关闭文档 → 退出 Word → 释放 COM 引用
        // ------------------------------------------------------------------
        private void MainForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            // 1) 先阻断后续任何对 Word 的 COM 访问（防止 Resize / 回调查再次触发 RPC）
            _embedded = false;

            // 2) 还原 Word 自己的窗口：只走 Win32 API，不含 RPC，无 0x800706BE 风险
            if (_wordHwnd != IntPtr.Zero)
            {
                try
                {
                    int style = NativeMethods.GetWindowStyle(_wordHwnd);
                    style &= ~NativeMethods.WS_CHILD;
                    style |= NativeMethods.WS_CAPTION | NativeMethods.WS_THICKFRAME |
                             NativeMethods.WS_SYSMENU | NativeMethods.WS_MINIMIZEBOX |
                             NativeMethods.WS_MAXIMIZEBOX;
                    NativeMethods.SetWindowStyle(_wordHwnd, style);
                    NativeMethods.SetParent(_wordHwnd, IntPtr.Zero);
                }
                catch (COMException) { /* hWnd 可能已失效，忽略 */ }
                _wordHwnd = IntPtr.Zero;
            }

            // 3) 直接退出 Word。Quit(0) 本身就会关闭全部文档且不保存，
            //    无需逐个 Documents.Item(i).Close()（那会反复换新代理、放大出错面）。
            //    Word 退出时 RPC 通道会断开，在此一次性吞掉 0x800706BE。
            if (_wordApp != null)
            {
                try { _wordApp.Quit(0); }
                catch (COMException) { /* Word 已自行退出 / 连接已断，忽略 */ }

                // 4) 释放 COM 代理。注意：Word 退出时释放代理也可能再抛，
                //    必须同层吞噬，不能放进 finally（finally 里的异常会覆盖上面的 catch）。
                try { Marshal.FinalReleaseComObject(_wordApp); }
                catch (COMException) { }
                _wordApp = null;
            }

            // 5) 不再手动 GC.Collect / WaitForPendingFinalizers：
            //    它会在窗体关闭期间强制回收全部遗留 COM 代理，此刻正向 Word 连接，
            //    极易再次命中 0x800706BE。交给运行时自然回收即可。
        }

        // ------------------------------------------------------------------
        // 工具方法
        // ------------------------------------------------------------------
        private void EnsureWordAlive()
        {
            if (_wordApp == null)
                throw new InvalidOperationException("Word 尚未启动或已被关闭。");
        }

        private void ShowComError(COMException ex)
        {
            MessageBox.Show("操作失败：" + ex.Message, "错误",
                MessageBoxButtons.OK, MessageBoxIcon.Warning);
        }

        private void SetStatus(string text)
        {
            _lblStatus.Text = text;
            _status.Refresh();
        }
    }
}
