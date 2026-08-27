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
            Text = AssemblyInfo.FRIENDLY_APP_NAME + " v" + AssemblyInfo.PRODUCT_VERSION
                   + " - Word 隔离嵌入示例 (.NET Framework 4.8)";
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
            _miPaste = new ToolStripMenuItem("粘贴(&P)", null, (s, e) =>
            {
                // 等菜单关闭后再粘贴，避免焦点仍停在 ToolStrip 上
                BeginInvoke(new Action(() => _host.Paste()));
            });
            // 真正绑定快捷键（菜单文本会自动显示 Ctrl+V），避免“写了却不生效”
            _miPaste.ShortcutKeys = Keys.Control | Keys.V;
            _menuEdit.DropDownItems.Add(_miPaste);

            _menu.Items.Add(_menuFile);
            _menu.Items.Add(_menuEdit);

            _status = new StatusStrip();
            _lblStatus = new ToolStripStatusLabel("就绪");
            _status.Items.Add(_lblStatus);

            _host = new WordProcessHost();
            // 控件不直接弹框/改状态栏，统一由窗体处理
            _host.HostError += OnHostError;
            _host.HostStateChanged += OnHostStateChanged;

            Controls.Add(_host);
            Controls.Add(_status);
            Controls.Add(_menu);
            MainMenuStrip = _menu;
        }

        private async void MainForm_Load(object sender, EventArgs e)
        {
            SetStatus("正在启动独立 Word 进程…");
            if (await _host.StartAsync())
                SetStatus("已嵌入独立 Word 进程；可直接编辑，Ctrl+C/Ctrl+V 可用。");
            else
                SetStatus("嵌入失败：详见日志。");
        }

        private async void NewDocument()
        {
            SetStatus("正在新建空白 Word 文档…");
            if (await _host.NewDocumentAsync())
                SetStatus("已新建空白 Word 文档。");
            else
                SetStatus("新建失败：详见日志。");
        }

        /// <summary>宿主控件报错上抛：由窗体统一弹框（图标从原来的“警告”统一为“错误”）。</summary>
        private void OnHostError(string msg)
        {
            MessageBox.Show(msg, AssemblyInfo.FRIENDLY_APP_NAME,
                MessageBoxButtons.OK, MessageBoxIcon.Error);
        }

        /// <summary>宿主状态变化：刷新状态栏；Word 自行退出时禁用粘贴。</summary>
        private void OnHostStateChanged(string text)
        {
            SetStatus(text);
            _miPaste.Enabled = text != "Word 进程已退出";
        }

        private void MainForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            _host.Stop();
        }

        private void SetStatus(string text)
        {
            _lblStatus.Text = text;
        }
    }
}
