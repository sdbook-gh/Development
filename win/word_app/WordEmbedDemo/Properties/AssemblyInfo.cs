// AssemblyInfo.cs -- project metadata for WordEmbedDemo.
// This file is referenced by build.cmd and WordEmbedDemo.csproj for csc
// (one-file compile) and msbuild respectively.

using System.Reflection;
using System.Runtime.InteropServices;

// 真正的程序集级元数据（之前只有字段、无任何 [assembly:] 特性，
// 导致 exe 属性页 / 任务管理器 / 客提报表里看到的是默认值）
[assembly: AssemblyTitle("WordEmbedDemo")]
[assembly: AssemblyDescription("以独立进程 + 窗口挂靠方式嵌入本机 Word 的示例")]
[assembly: AssemblyConfiguration("")]
[assembly: AssemblyCompany("")]
[assembly: AssemblyProduct("WordEmbedDemo")]
[assembly: AssemblyCopyright("Copyright © 2025")]
[assembly: AssemblyTrademark("")]
[assembly: AssemblyCulture("")]
[assembly: ComVisible(false)]
[assembly: Guid("a1f3c2d4-5b6e-47a8-9c0d-1e2f3a4b5c6d")]
[assembly: AssemblyVersion("1.0.0.0")]
[assembly: AssemblyFileVersion("1.0.0.0")]

namespace WordEmbedDemo
{
    internal static class AssemblyInfo
    {
        /// <summary>
        /// 显示给用户的应用名称。
        /// </summary>
        public static readonly string FRIENDLY_APP_NAME = "WordEmbedDemo";

        /// <summary>
        /// 版本号（主版本.次版本.修订）。
        /// </summary>
        public static readonly string PRODUCT_VERSION = "1.0.0";

        /// <summary>
        /// 内部标题，用于窗口标题等。
        /// </summary>
        public static readonly string INTERNAL_NAME = "WordEmbedDemo";

        /// <summary>
        /// 版权信息。
        /// </summary>
        public static readonly string COPYRIGHT = "Copyright (C) 2025";
    }
}
