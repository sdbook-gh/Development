using System;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace WordEmbedDemo
{
    // =====================================================================
    //  OleWordHost
    //  ----------
    //  把一个 Word.Document **OLE 嵌入对象** 以“就地激活”(in-place
    //  activation) 方式嵌进本控件窗口内，在本面板内就地编辑。
    //
    //  与旧实现的本质区别：旧实现抓全局唯一 Word.Application 再 SetParent，
    //  导致另开 Word 连到同一实例、互相串扰。本实现嵌入单个文档对象，由
    //  Word 自己的 OLE 本地服务器进程托管、仅在“本面板”内就地编辑——
    //  与其它 Word 窗口完全隔离。
    // =====================================================================

    #region 结构
    [StructLayout(LayoutKind.Sequential)]
    public struct MyRECT
    {
        public int Left, Top, Right, Bottom;
        public MyRECT(int l, int t, int r, int b) { Left = l; Top = t; Right = r; Bottom = b; }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct MySIZEL { public int cx, cy; }

    [StructLayout(LayoutKind.Sequential)]
    public struct MyBorder { public int Left, Top, Right, Bottom; }

    [StructLayout(LayoutKind.Sequential)]
    public struct MyFormatEtc
    {
        public short cfFormat;
        public IntPtr ptd;
        public uint dwAspect;
        public int lindex;
        public uint tymed;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct MyFrameInfo
    {
        public uint cb;
        public int fMDIApp;
        public IntPtr hwndFrame;
        public IntPtr haccel;
        public uint cAccelEntries;
    }
    #endregion

    #region P/Invoke + 常量
    internal static class OleNative
    {
        public static Guid IID_IOleObject = new Guid("00000112-0000-0000-C000-000000000046");
        public const uint INPLACEACTIVATE = 1;
        public const uint ASPECT_CONTENT = 1;
        public const uint STGM_READWRITE = 0x2;
        public const uint STGM_EXCL = 0x10;
        public const uint STGM_CREATE = 0x1000;

        [DllImport("ole32.dll")] public static extern int OleInitialize(IntPtr p);
        [DllImport("ole32.dll")] public static extern void OleUninitialize();
        [DllImport("ole32.dll", CharSet = CharSet.Unicode)]
        public static extern int CLSIDFromProgID(string p, out Guid c);
        [DllImport("ole32.dll")]
        public static extern int CreateILockBytesOnHGlobal(IntPtr h, int del, out IntPtr lb);
        [DllImport("ole32.dll")]
        public static extern int StgCreateDocfileOnILockBytes(IntPtr lb, uint f, int r, out IntPtr st);
        [DllImport("ole32.dll")]
        public static extern int OleCreate(ref Guid cl, ref Guid riid, IntPtr pUnkOuter,
            uint renderopt, ref MyFormatEtc pFormatetc, IntPtr pClientSite, IntPtr pStg, out IntPtr o);
        [DllImport("ole32.dll")] public static extern int OleRun(IntPtr o);
        [DllImport("ole32.dll")] public static extern int OleSetContainedObject(IntPtr o, int f);

        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        public static extern IntPtr SetFocus(IntPtr h);
    }
    #endregion

    #region 对象端接口 IOleObject（宿主调用端）
    [ComImport, ComVisible(true)]
    [Guid("00000112-0000-0000-C000-000000000046")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IOleObjectIfc
    {
        void SetClientSite(object site);
        void GetClientSite(out object site);
        void SetHostNames(string a, string b);
        void Close(int save);
        void SetMoniker(uint w, IntPtr m);
        void GetMoniker(uint wa, uint ww, out IntPtr m);
        void InitFromData(IntPtr d, int f, int r);
        void GetClipboardData(int r, out IntPtr d);
        void DoVerb(uint v, IntPtr msg, object site, uint l, IntPtr h, ref MyRECT rc);
        void EnumVerbs(out object e);
        void OleUpdate();
        void IsUpToDate();
        void GetUserClassID(out Guid g);
        void GetUserType(uint f, out IntPtr s);
        void SetExtent(uint a, ref MySIZEL s);
        void GetExtent(uint a, ref MySIZEL s);
        void Advise(object s, out uint c);
        void Unadvise(uint c);
        void EnumAdvise(out object e);
        void GetMiscStatus(uint a, out uint s);
        void SetColorScheme(IntPtr p);
    }
    #endregion

    #region 宿主需实现的就地接口
    [ComImport, ComVisible(true)]
    [Guid("00000118-0000-0000-C000-000000000046")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IOleClientSiteIfc
    {
        void SaveObject();
        void GetMoniker(uint dwAssign, uint dwWhich, out IntPtr ppmk);
        void GetContainer(out IOleContainerIfc pp);
        void ShowObject();
        void OnShowWindow([MarshalAs(UnmanagedType.Bool)] bool f);
        void RequestNewObjectLayout();
    }

    [ComImport, ComVisible(true)]
    [Guid("0000011B-0000-0000-C000-000000000046")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IOleContainerIfc
    {
        void ParseDisplayName(IntPtr pbc, string n, out uint eaten, out IntPtr mon);
        void EnumObjects(uint grf, out IntPtr p);
        void LockContainer([MarshalAs(UnmanagedType.Bool)] bool f);
        void UnlockContainer();
    }

    [ComImport, ComVisible(true)]
    [Guid("00000114-0000-0000-C000-000000000046")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IOleWindowIfc
    {
        void GetWindow(out IntPtr h);
        void ContextSensitiveHelp([MarshalAs(UnmanagedType.Bool)] bool f);
    }

    [ComImport, ComVisible(true)]
    [Guid("00000119-0000-0000-C000-000000000046")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IOleInPlaceSiteIfc
    {
        void GetWindow(out IntPtr h);
        void ContextSensitiveHelp([MarshalAs(UnmanagedType.Bool)] bool f);
        void CanInPlaceActivate();
        void OnInPlaceActivate();
        void OnUIActivate();
        void GetWindowContext(out IOleInPlaceFrameIfc f, out IOleInPlaceUIWindowIfc d,
                              ref MyRECT pos, ref MyRECT clip, ref MyFrameInfo info);
        void Scroll(IntPtr p);
        void OnUIDeactivate([MarshalAs(UnmanagedType.Bool)] bool f);
        void OnUINeverDeactivate();
        void OnInPlaceDeactivate();
        void DiscardUndoState();
        void DeactivateAndUndo();
        void OnPosRectChange(ref MyRECT rc);
    }

    [ComImport, ComVisible(true)]
    [Guid("00000115-0000-0000-C000-000000000046")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IOleInPlaceUIWindowIfc
    {
        void GetWindow(out IntPtr h);
        void ContextSensitiveHelp([MarshalAs(UnmanagedType.Bool)] bool f);
        void GetBorder(out MyRECT rc);
        void RequestBorderSpace(ref MyBorder b);
        void SetBorderSpace(ref MyBorder b);
        void SetActiveObject(IntPtr p, string n);
    }

    [ComImport, ComVisible(true)]
    [Guid("00000116-0000-0000-C000-000000000046")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IOleInPlaceFrameIfc
    {
        void GetWindow(out IntPtr h);
        void ContextSensitiveHelp([MarshalAs(UnmanagedType.Bool)] bool f);
        void GetBorder(out MyRECT rc);
        void RequestBorderSpace(ref MyBorder b);
        void SetBorderSpace(ref MyBorder b);
        void SetActiveObject(IntPtr p, string n);
        int InsertMenus(IntPtr h, IntPtr w);
        int SetMenu(IntPtr h, IntPtr o, IntPtr a);
        void RemoveMenus(IntPtr h);
        int SetStatusText(string s);
        void EnableModeless([MarshalAs(UnmanagedType.Bool)] bool f);
        int TranslateAccelerator(ref int m, ushort w);
    }
    #endregion

    // =====================================================================
    [ComVisible(true)]
    public class OleWordHost : Control,
        IOleClientSiteIfc, IOleContainerIfc, IOleWindowIfc, IOleInPlaceSiteIfc,
        IOleInPlaceUIWindowIfc, IOleInPlaceFrameIfc
    {
        private IOleObjectIfc _obj;
        private IntPtr _pObj;
        private IntPtr _storage;
        private static bool _init;
        private bool _run;

        public OleWordHost() { Dock = DockStyle.Fill; BackColor = Color.White; }

        private static void Trace(string msg)
        {
            try
            {
                System.IO.File.AppendAllText(
                    System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ole_trace.txt"),
                    DateTime.Now.ToString("HH:mm:ss.fff") + "  " + msg + Environment.NewLine);
            }
            catch { }
        }

        public bool Start()
        {
            try
            {
                Trace("Start begin");
                if (!_init) { OleNative.OleInitialize(IntPtr.Zero); _init = true; }

                Guid clsid;
                if (OleNative.CLSIDFromProgID("Word.Document", out clsid) < 0) return Fail("无法解析 Word.Document");

                IntPtr lb;
                if (OleNative.CreateILockBytesOnHGlobal(IntPtr.Zero, 1, out lb) < 0) return Fail("创建 ILockBytes 失败");

                if (OleNative.StgCreateDocfileOnILockBytes(lb, OleNative.STGM_READWRITE | OleNative.STGM_EXCL | OleNative.STGM_CREATE, 0, out _storage) < 0)
                    return Fail("创建 Storage 失败");

                IntPtr cs = Marshal.GetIUnknownForObject(this);

                // FORMATETC：OLERENDER_DRAW，默认 TYMED_NULL
                MyFormatEtc fe = new MyFormatEtc();
                fe.cfFormat = 0;
                fe.ptd = IntPtr.Zero;
                fe.dwAspect = OleNative.ASPECT_CONTENT;
                fe.lindex = -1;
                fe.tymed = 0; // TYMED_NULL

                IntPtr p;
                int hr = OleNative.OleCreate(ref clsid, ref OleNative.IID_IOleObject, IntPtr.Zero,
                    1 /*OLERENDER_DRAW*/, ref fe, cs, _storage, out p);
                Marshal.Release(cs);
                if (hr < 0) return Fail("OleCreate 失败 0x" + hr.ToString("X8"));
                Trace("OleCreate ok");

                _pObj = p;
                _obj = (IOleObjectIfc)Marshal.GetObjectForIUnknown(p);
                _obj.SetClientSite(this);
                OleNative.OleSetContainedObject(_pObj, 1);
                OleNative.OleRun(_pObj);
                _obj.SetHostNames("WordEmbedDemo", "Word");

                MySIZEL sz = new MySIZEL { cx = ClientSize.Width, cy = ClientSize.Height };
                _obj.SetExtent(OleNative.ASPECT_CONTENT, ref sz);
                Trace("OleCreate + setup ok");

                MyRECT rc = new MyRECT(0, 0, ClientSize.Width, ClientSize.Height);
                _obj.DoVerb(OleNative.INPLACEACTIVATE, IntPtr.Zero, this, 0, Handle, ref rc);

                _run = true;
                PositionChild();
                Trace("Start end (success)");
                return true;
            }
            catch (COMException ex) { Trace("COMException: " + ex.Message); return Fail(ex.Message); }
            catch (Exception ex) { Trace("Exception: " + ex); return Fail(ex.Message); }
        }

        private bool Fail(string m)
        {
            MessageBox.Show(m, "嵌入失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return false;
        }

        public void PositionChild()
        {
            if (_obj == null) return;
            IntPtr c = NativeMethods.FindWindowEx(Handle, IntPtr.Zero, null, null);
            if (c != IntPtr.Zero)
                NativeMethods.MoveWindow(c, 0, 0, Math.Max(1, ClientSize.Width), Math.Max(1, ClientSize.Height), true);
        }

        protected override void OnResize(EventArgs e)
        {
            base.OnResize(e);
            if (_run) PositionChild();
        }

        /// <summary>聚焦嵌入文档并执行 Ctrl+V 粘贴。</summary>
        public void Paste()
        {
            IntPtr c = NativeMethods.FindWindowEx(Handle, IntPtr.Zero, null, null);
            if (c == IntPtr.Zero) return;
            OleNative.SetFocus(c);
            SendKeys.SendWait("^v");
        }

        /// <summary>丢弃当前对象并重新嵌入一个新的空白 Word 文档。</summary>
        public void NewDocument()
        {
            StopEmbedding();
            Start();
        }

        public void StopEmbedding()
        {
            if (_obj != null)
            {
                try { _obj.Close(1); } catch (Exception) { }
                try { OleNative.OleSetContainedObject(_pObj, 0); } catch (Exception) { }
                try { Marshal.ReleaseComObject(_obj); } catch (Exception) { }
                _obj = null;
            }
            if (_pObj != IntPtr.Zero)
            {
                try { Marshal.Release(_pObj); } catch (Exception) { }
                _pObj = IntPtr.Zero;
            }
            if (_storage != IntPtr.Zero)
            {
                try { Marshal.Release(_storage); } catch (Exception) { }
                _storage = IntPtr.Zero;
            }
            _run = false;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing) StopEmbedding();
            base.Dispose(disposing);
        }

        #region 宿主接口实现（空实现 = S_OK；就地编辑所需）
        public void SaveObject() { }
        public void GetMoniker(uint a, uint b, out IntPtr c) { c = IntPtr.Zero; }
        public void GetContainer(out IOleContainerIfc pp) { pp = this; }
        public void ShowObject() { }
        public void OnShowWindow(bool f) { }
        public void RequestNewObjectLayout() { }

        public void ParseDisplayName(IntPtr pbc, string n, out uint e, out IntPtr m) { e = 0; m = IntPtr.Zero; }
        public void EnumObjects(uint g, out IntPtr p) { p = IntPtr.Zero; }
        public void LockContainer(bool f) { }
        public void UnlockContainer() { }

        public void GetWindow(out IntPtr h) { h = Handle; }
        public void ContextSensitiveHelp(bool f) { }

        public void CanInPlaceActivate() { }
        public void OnInPlaceActivate() { }
        public void OnUIActivate() { }
        public void GetWindowContext(out IOleInPlaceFrameIfc f, out IOleInPlaceUIWindowIfc d,
                                     ref MyRECT pos, ref MyRECT clip, ref MyFrameInfo info)
        {
            f = this; d = this;
            pos = new MyRECT(0, 0, Math.Max(1, ClientSize.Width), Math.Max(1, ClientSize.Height));
            clip = new MyRECT(0, 0, Math.Max(1, ClientSize.Width), Math.Max(1, ClientSize.Height));
            info.cb = (uint)Marshal.SizeOf(typeof(MyFrameInfo));
            info.fMDIApp = 0;
            info.hwndFrame = Handle;
            info.haccel = IntPtr.Zero;
            info.cAccelEntries = 0;
        }
        public void Scroll(IntPtr p) { }
        public void OnUIDeactivate(bool f) { }
        public void OnUINeverDeactivate() { }
        public void OnInPlaceDeactivate() { }
        public void DiscardUndoState() { }
        public void DeactivateAndUndo() { }
        public void OnPosRectChange(ref MyRECT rc) { }

        // ---- IOleInPlaceUIWindowIfc / IOleInPlaceFrameIfc（就地宿主，不提供菜单/状态栏）----
        public void GetBorder(out MyRECT rc) { rc = new MyRECT(0, 0, Math.Max(1, ClientSize.Width), Math.Max(1, ClientSize.Height)); }
        public void RequestBorderSpace(ref MyBorder b) { }
        public void SetBorderSpace(ref MyBorder b) { }
        public void SetActiveObject(IntPtr p, string n) { }
        public int InsertMenus(IntPtr h, IntPtr w) { return unchecked((int)0x80004001); }        // E_NOTIMPL：不往 Word 塞菜单
        public int SetMenu(IntPtr h, IntPtr o, IntPtr a) { return unchecked((int)0x80004001); }
        public void RemoveMenus(IntPtr h) { }
        public int SetStatusText(string s) { return unchecked((int)0x80004001); }
        public void EnableModeless(bool f) { }
        public int TranslateAccelerator(ref int m, ushort w) { return 0; }        // 不处理加速键
        #endregion
    }
}