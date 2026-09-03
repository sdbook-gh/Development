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
        private ToolStripMenuItem _miExit;
        private ToolStripMenuItem _menuEdit;
        private ToolStripMenuItem _miPaste;
        private StatusStrip _status;
        private ToolStripStatusLabel _lblStatus;
        private WordProcessHost _host;
        private bool _closing;   // FormClosing 后 Load 续流不再弹「嵌入失败」

        public MainForm()
        {
            Text = AssemblyInfo.FRIENDLY_APP_NAME + " v" + AssemblyInfo.PRODUCT_VERSION
                   + " - Word 隔离嵌入示例 (.NET Framework 4.8)";
            Width = 1000;
            Height = 720;
            StartPosition = FormStartPosition.CenterScreen;

            BuildUi();

            Load += MainForm_Load;
            FormClosing += MainForm_FormClosing;
            Resize += MainForm_Resize;
        }

        private void BuildUi()
        {
            _menu = new MenuStrip();
            _menuFile = new ToolStripMenuItem("文件(&F)");
            _miExit = new ToolStripMenuItem("退出(&X)", null, (s, e) => Close());
            _menuFile.DropDownItems.Add(_miExit);

            _menuEdit = new ToolStripMenuItem("编辑(&E)");
            _miPaste = new ToolStripMenuItem("粘贴(&P)  Ctrl+V", null, (s, e) =>
            {
                // 等菜单关闭后再粘贴，避免焦点仍停在 ToolStrip 上
                BeginInvoke(new Action(() => _host.Paste()));
            });
            _menuEdit.DropDownItems.Add(_miPaste);

            _menu.Items.Add(_menuFile);
            _menu.Items.Add(_menuEdit);

            _status = new StatusStrip();
            _lblStatus = new ToolStripStatusLabel("就绪");
            _status.Items.Add(_lblStatus);

            _host = new WordProcessHost();
            // 控件不直接弹框，统一由窗体处理
            _host.HostError += OnHostError;
            _host.HostStateChanged += OnHostStateChanged;

            Controls.Add(_menu);
            Controls.Add(_status);
            Controls.Add(_host);
            MainMenuStrip = _menu;
            // 初始居中布局
            CenterHost();
        }

        private async void MainForm_Load(object sender, EventArgs e)
        {
            try
            {
                bool ok = await _host.StartAsync();
                if (_closing || IsCancelledError(_host.LastError))
                    return;
                if (!ok)
                    ShowStartFailure(_host.LastError);
            }
            catch (OperationCanceledException)
            {
                if (_closing) return;
                SetStatus("启动已取消。");
            }
            catch (Exception ex)
            {
                if (_closing || IsCancelledError(_host.LastError)) return;
                string msg = "Word 嵌入启动失败：" + ex.Message;
                SetStatus(msg);
                // 仅当窗体仍存活时弹框；若启动流程本身已走 HostError，此处作为兜底保护。
                if (!IsDisposed && IsHandleCreated)
                    ShowErrorBox(msg);
            }
        }

        /// <summary>宿主控件报错上抛：由窗体统一弹框（图标从原来的“警告”统一为“错误”）。</summary>
        private void OnHostError(string msg)
        {
            if (_closing || IsCancelledError(msg)) return;
            ShowErrorBox(msg);
        }

        /// <summary>
        /// 启动失败兜底弹框。找不到 WINWORD 时用明确文案；
        /// 若 HostError 已经弹过同一句话，ShowErrorBox 仍再弹一次也可以接受，
        /// 因此用 _startErrorShown 去重。
        /// </summary>
        private bool _startErrorShown;

        private static bool IsCancelledError(string detail)
        {
            return detail == WordProcessHost.CancelledSentinel;
        }

        private void ShowStartFailure(string detail)
        {
            if (_closing || IsCancelledError(detail)) return;
            if (_startErrorShown) return;
            bool notFound = !string.IsNullOrEmpty(detail) &&
                (detail.IndexOf("WINWORD", StringComparison.OrdinalIgnoreCase) >= 0
                 || detail.IndexOf("未找到") >= 0);
            string msg;
            if (notFound)
            {
                msg = "未找到 Microsoft Word。请确认已安装 Word 后重试。";
                if (!string.IsNullOrEmpty(detail))
                    msg = msg + Environment.NewLine + Environment.NewLine + detail;
            }
            else if (!string.IsNullOrEmpty(detail))
                msg = detail;
            else
                msg = "嵌入 Word 失败。";
            ShowErrorBox(msg);
        }

        private void ShowErrorBox(string msg)
        {
            if (_closing || IsCancelledError(msg)) return;
            if (string.IsNullOrEmpty(msg)) return;
            if (IsDisposed || !IsHandleCreated) return;
            _startErrorShown = true;
            MessageBox.Show(this, msg, AssemblyInfo.FRIENDLY_APP_NAME,
                MessageBoxButtons.OK, MessageBoxIcon.Error);
        }

        /// <summary>宿主状态变化：按状态刷新状态栏与粘贴菜单可用性。</summary>
        private void OnHostStateChanged(WordProcessHost.HostStatus status)
        {
            switch (status)
            {
                case WordProcessHost.HostStatus.Running:
                    _miPaste.Enabled = true;
                    SetStatus("已嵌入独立 Word 进程；可直接编辑，Ctrl+C/Ctrl+V 可用。");
                    break;
                case WordProcessHost.HostStatus.Exited:
                    _miPaste.Enabled = false;
                    SetStatus("Word 进程已退出");
                    break;
                case WordProcessHost.HostStatus.Failed:
                    _miPaste.Enabled = false;
                    SetStatus("嵌入失败：详见日志。");
                    break;
            }
        }

        private void MainForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            _closing = true;
            // 必须走还原 + Quit + 超时 Kill：不能 forceKill 跳过还原，
            // 否则用户自己的 Word 会留下无 Ribbon / 无状态栏。
            _host.Stop(forceKill: false);
        }

        /// <summary>让内嵌 Word 宿主控件在窗体中央居中显示（四周各留 5% 边距），并避开顶部菜单栏与底部状态栏。</summary>
        private void CenterHost()
        {
            if (_host == null) return;
            int menuHeight = _menu != null ? _menu.Height : 0;        // 顶部菜单栏高度
            int statusHeight = _status != null ? _status.Height : 0;  // 底部状态栏高度
            int availW = ClientSize.Width;
            int availH = Math.Max(1, ClientSize.Height - menuHeight - statusHeight);
            int marginW = Math.Max(1, availW / 20);    // 5% 边距
            int marginH = Math.Max(1, availH / 20);
            int w = Math.Max(1, availW - 2 * marginW);
            int h = Math.Max(1, availH - 2 * marginH);
            _host.Location = new System.Drawing.Point(marginW, menuHeight + marginH);
            _host.Size = new System.Drawing.Size(w, h);
        }

        /// <summary>窗体大小变化时重新居中，保证内嵌 Word 始终位于窗体中心。</summary>
        private void MainForm_Resize(object sender, EventArgs e)
        {
            CenterHost();
        }

        /// <summary>更新状态栏文本。</summary>
        private void SetStatus(string text)
        {
            if (_lblStatus == null || _status == null) return;
            _lblStatus.Text = text;
            _status.Refresh();
        }
    }
}
