word_app 安装说明
====================

本项目使用 Python 虚拟环境 (venv) 隔离依赖，依赖版本通过 version.toml 管理。

------------------------------------------------------------
1. 前置要求
------------------------------------------------------------
- Python 3.9 或更高版本（推荐 3.11+）
- pip（随 Python 自带）
- 操作系统：Windows 10/11（项目用到 pywin32）

确认 Python 已安装：
    python --version

------------------------------------------------------------
2. 创建并激活虚拟环境 (venv)
------------------------------------------------------------
在项目根目录（word_app）下执行：

创建虚拟环境：
    python -m venv venv

激活虚拟环境（Windows PowerShell）：
    .\venv\Scripts\Activate.ps1

激活虚拟环境（Windows CMD）：
    .\venv\Scripts\activate.bat

激活成功后，命令行前会出现 (venv) 前缀。
如需退出虚拟环境，执行：
    deactivate

------------------------------------------------------------
3. 升级 pip（可选但推荐）
------------------------------------------------------------
    python -m pip install --upgrade pip

------------------------------------------------------------
4. 使用 version.toml 安装依赖
------------------------------------------------------------
version.toml 采用 TOML 格式记录依赖及其版本，例如：

    [dependencies]
    PySide6 = "6.11.2"
    pywin32 = "312"
    python-docx = "1.2.0"
    html-for-docx = "1.2.0"

由于标准 pip 不直接读取 TOML，可使用以下任一方式安装。

方式 A：手动转换为 pip 命令（最简单，无需额外工具）
    pip install PySide6==6.11.2 pywin32==312 python-docx==1.2.0 html-for-docx==1.2.0

方式 B：使用 tomllib（Python 3.11+ 内置）解析后安装
在项目根目录执行：
    python -c "import tomllib, subprocess, sys; d=tomllib.load(open('version.toml','rb'))['dependencies']; subprocess.check_call([sys.executable,'-m','pip','install']+[f'{k}=={v}' for k,v in d.items()])"

方式 C：使用 uv（更现代，原生支持 TOML）
    pip install uv
    uv pip install -r version.toml

------------------------------------------------------------
5. 验证安装
------------------------------------------------------------
    python -c "import PySide6, win32api, docx; print('OK')"

------------------------------------------------------------
6. 运行项目
------------------------------------------------------------
    python main.py

------------------------------------------------------------
7. 常见问题
------------------------------------------------------------
- PowerShell 执行策略导致激活脚本被禁止运行：
    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  然后重新执行激活命令。

- pywin32 安装后仍报 ImportError：
    python .\venv\Scripts\pywin32_postinstall.py -install

- 需要新增依赖时，先 pip install <包>，再把包名和版本写入 version.toml 的 [dependencies] 段，保持版本一致。
