using System;
using System.Windows.Forms;

namespace WordEmbedDemo
{
    /// <summary>
    /// 主窗体：以“独立进程 + 窗口挂靠”方式嵌入本机 Word，
    /// 且只操作我们自己新创建的 Word 进程（PID 锁定），
    /// 与用户单独打开的其它 Word 完全隔离互不影响。
    /// </summary>
    public class MainForm : Form
    {
        private MenuStrip _menu;
        private ToolStripMenuItem _menuFile;
        private ToolStripMenuItem _miNew;
        private ToolStripMenuItem _miExit;
        private ToolStripMenuItem _menuEdit;
        private ToolStripMenuItem _miPaste;
        private WordProcessHost _host;
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

            _host = new WordProcessHost();

            Controls.Add(_host);
            Controls.Add(_status);
            Controls.Add(_menu);
            MainMenuStrip = _menu;
        }

        private void MainForm_Load(object sender, EventArgs e)
        {
            if (_host.Start())
                SetStatus("已嵌入独立 Word 进程；可直接编辑，Ctrl+C/Ctrl+V 可用。");
            else
                SetStatus("嵌入失败：详见日志。");
        }

        private void NewDocument()
        {
            _host.NewDocument();
            SetStatus("已新建空白 Word 文档。");
        }

        private void MainForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            _host.Stop();
        }

        private void SetStatus(string text)
        {
            _lblStatus.Text = text;
            _status.Refresh();
        }
    }
}
