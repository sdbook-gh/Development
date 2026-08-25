using System;
using System.Windows.Forms;

namespace WordEmbedDemo
{
    /// <summary>
    /// 主窗体：把单个 Word.Document **OLE 嵌入对象**就地编辑在本窗体面板里。
    /// 与旧实现不同：这里不再抓取全局唯一的 Word.Application 并 SetParent 其
    /// 顶层窗口 —— 因此你另开 Word 与本应用互不干扰。
    /// </summary>
    public class MainForm : Form
    {
        private MenuStrip _menu;
        private ToolStripMenuItem _menuFile;
        private ToolStripMenuItem _miNew;
        private ToolStripMenuItem _miExit;
        private ToolStripMenuItem _menuEdit;
        private ToolStripMenuItem _miPaste;
        private OleWordHost _host;
        private StatusStrip _status;
        private ToolStripStatusLabel _lblStatus;

        public MainForm()
        {
            Text = "Word 隔离嵌入示例 (.NET Framework 4.8)";
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
            _menu = new MenuStrip();
            _menuFile = new ToolStripMenuItem("文件(&F)");
            _miNew = new ToolStripMenuItem("新建文档(&N)", null, (s, e) => NewDocument());
            _miExit = new ToolStripMenuItem("退出(&X)", null, (s, e) => Close());
            _menuFile.DropDownItems.Add(_miNew);
            _menuFile.DropDownItems.Add(_miExit);

            _menuEdit = new ToolStripMenuItem("编辑(&E)");
            _miPaste = new ToolStripMenuItem("粘贴(&P)  Ctrl+V", null, (s, e) => _host.Paste());
            _menuEdit.DropDownItems.Add(_miPaste);

            _menu.Items.Add(_menuFile);
            _menu.Items.Add(_menuEdit);

            _status = new StatusStrip();
            _lblStatus = new ToolStripStatusLabel("就绪");
            _status.Items.Add(_lblStatus);

            _host = new OleWordHost();

            Controls.Add(_host);
            Controls.Add(_status);
            Controls.Add(_menu);
            MainMenuStrip = _menu;
        }

        private void MainForm_Load(object sender, EventArgs e)
        {
            if (_host.Start())
                SetStatus("已嵌入空白 Word 文档：直接编辑即可；Ctrl+C/Ctrl+V 均可用。");
            else
                SetStatus("嵌入失败。");
        }

        private void NewDocument()
        {
            _host.NewDocument();
            SetStatus("已新建空白 Word 文档。");
        }

        public void Paste()
        {
            _host.Paste();
        }

        private void MainForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            _host.StopEmbedding();
        }

        private void SetStatus(string text)
        {
            _lblStatus.Text = text;
            _status.Refresh();
        }
    }
}