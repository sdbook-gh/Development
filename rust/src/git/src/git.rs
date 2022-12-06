mod raw {
    #![allow(non_camel_case_types)]
    use std::os::raw::{c_char, c_int, c_uchar};

    #[link(name = "git2")]
    extern "C" {
        pub fn git_libgit2_init() -> c_int;
        pub fn git_libgit2_shutdown() -> c_int;
        pub fn giterr_last() -> *const git_error;
        pub fn git_repository_open(out: *mut *mut git_repository, path: *const c_char) -> c_int;
        pub fn git_repository_free(repo: *mut git_repository);
        pub fn git_reference_name_to_id(
            out: *mut git_oid,
            repo: *mut git_repository,
            reference: *const c_char,
        ) -> c_int;
        pub fn git_commit_lookup(
            out: *mut *mut git_commit,
            repo: *mut git_repository,
            id: *const git_oid,
        ) -> c_int;
        pub fn git_commit_author(commit: *const git_commit) -> *const git_signature;
        pub fn git_commit_message(commit: *const git_commit) -> *const c_char;
        pub fn git_commit_free(commit: *mut git_commit);
    }
    #[repr(C)]
    pub struct git_repository {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct git_commit {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct git_error {
        pub message: *const c_char,
        pub klass: c_int,
    }
    pub const GIT_OID_RAWSZ: usize = 20;
    #[repr(C)]
    pub struct git_oid {
        pub id: [c_uchar; GIT_OID_RAWSZ],
    }
    pub type git_time_t = i64;
    #[repr(C)]
    pub struct git_time {
        pub time: git_time_t,
        pub offset: c_int,
    }
    #[repr(C)]
    pub struct git_signature {
        pub name: *const c_char,
        pub email: *const c_char,
        pub when: git_time,
    }
}

use std::ffi::CStr;
use std::os::raw::c_int;

fn check(activity: &'static str, status: c_int) -> c_int {
    if status < 0 {
        unsafe {
            let error = &*raw::giterr_last();
            println!(
                "error while {}: {} ({})",
                activity,
                CStr::from_ptr(error.message).to_string_lossy(),
                error.klass
            );
            std::process::exit(1);
        }
    }
    status
}
unsafe fn show_commit(commit: *const raw::git_commit) {
    let author = raw::git_commit_author(commit);
    let name = CStr::from_ptr((*author).name).to_string_lossy();
    let email = CStr::from_ptr((*author).email).to_string_lossy();
    println!("{} <{}>\n", name, email);
    let message = raw::git_commit_message(commit);
    println!("{}", CStr::from_ptr(message).to_string_lossy());
}

fn test_git() {
    let path = std::ffi::CString::new("/mnt/d/personal/github/Development").unwrap();
    unsafe {
        check("initializing library", raw::git_libgit2_init());
        let mut repo = std::ptr::null_mut();
        check(
            "opening repository",
            raw::git_repository_open(&mut repo, path.as_ptr()),
        );
        let c_name = std::ffi::CString::new("HEAD").unwrap();
        let oid = {
            // let mut oid = std::mem::MaybeUninit::uninit();
            let mut oid = raw::git_oid {
                id: [0; raw::GIT_OID_RAWSZ],
            };
            check(
                "looking up HEAD",
                raw::git_reference_name_to_id(&mut oid, repo, c_name.as_ptr()),
            );
            // oid.assume_init()
            oid
        };
        let mut commit = std::ptr::null_mut();
        check(
            "looking up commit",
            raw::git_commit_lookup(&mut commit, repo, &oid),
        );
        show_commit(commit);
        raw::git_commit_free(commit);
        raw::git_repository_free(repo);
        check("shutting down library", raw::git_libgit2_shutdown());
    }
}

pub mod util {
    use std::error;
    use std::ffi::CString;
    use std::fmt;
    use std::result;
    #[derive(Debug)]
    pub struct Error {
        code: i32,
        message: String,
        class: i32,
    }
    impl fmt::Display for Error {
        fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
            // Display 一个`Error`只需要 display `libgit2`的错误信息
            self.message.fmt(f)
        }
    }
    impl error::Error for Error {}
    pub type Result<T> = result::Result<T, Error>;
    impl From<String> for Error {
        fn from(message: String) -> Error {
            Error {
                code: -1,
                message,
                class: 0,
            }
        }
    }
    // NulError是如果字符里有 0字节时`CString::new`会返回的 Error类型
    impl From<std::ffi::NulError> for Error {
        fn from(e: std::ffi::NulError) -> Error {
            Error {
                code: -1,
                message: e.to_string(),
                class: 0,
            }
        }
    }
    use super::raw;
    use std::ffi::CStr;
    use std::os::raw::c_int;
    fn check(code: c_int) -> Result<c_int> {
        if code >= 0 {
            return Ok(code);
        }
        unsafe {
            let error = raw::giterr_last();
            // libgit2保证 (*error).message总是非空并且以空字符结尾
            // 所以这里的调用是安全的。
            let message = CStr::from_ptr((*error).message)
                .to_string_lossy()
                .into_owned();
            Err(Error {
                code: code as i32,
                message,
                class: (*error).klass as i32,
            })
        }
    }
    /// 一个 Git 仓库
    pub struct Repository {
        // 它必须总是指向一个还在生存的`git_repository`结构体，
        // 不能有别的`Repository`也指向同一个结构体。
        raw: *mut raw::git_repository,
    }
    use std::path::Path;
    use std::ptr;
    impl Repository {
        pub fn open(path: &str) -> Result<Repository> {
            ensure_initialized();
            let path = path_to_cstring(path.as_ref())?;
            let mut repo = ptr::null_mut();
            unsafe {
                check(raw::git_repository_open(&mut repo, path.as_ptr()))?;
            }
            Ok(Repository { raw: repo })
        }
    }
    impl Drop for Repository {
        fn drop(&mut self) {
            unsafe {
                raw::git_repository_free(self.raw);
            }
        }
    }
    fn ensure_initialized() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| unsafe {
            check(raw::git_libgit2_init()).expect("initializing libgit2 failed");
            assert_eq!(libc::atexit(shutdown), 0);
        });
    }
    extern "C" fn shutdown() {
        unsafe {
            if let Err(e) = check(raw::git_libgit2_shutdown()) {
                eprintln!("shutting down libgit2 failed: {}", e);
                std::process::abort();
            }
        }
    }
    #[cfg(unix)]
    fn path_to_cstring(path: &Path) -> Result<CString> {
        // `as_bytes`方法只在类 Unix系统中存在。
        use std::os::unix::ffi::OsStrExt;
        Ok(CString::new(path.as_os_str().as_bytes())?)
    }
    #[cfg(windows)]
    fn path_to_cstring(path: &Path) -> Result<CString> {
        // 尝试转换为 UTF-8。如果失败了，libgit2就不能处理这个路径了。
        match path.to_str() {
            Some(s) => Ok(CString::new(s)?),
            None => {
                let message = format!("Couldn't convert path '{}' to UTF-8", path.display());
                Err(message.into())
            }
        }
    }
    pub struct Oid {
        pub raw: raw::git_oid,
    }
    impl Repository {
        pub fn reference_name_to_id(&self, name: &str) -> Result<Oid> {
            let name = CString::new(name)?;
            unsafe {
                let oid = {
                    let mut oid = std::mem::MaybeUninit::uninit();
                    check(raw::git_reference_name_to_id(
                        oid.as_mut_ptr(),
                        self.raw,
                        name.as_ptr(),
                    ))?;
                    oid.assume_init()
                };
                Ok(Oid { raw: oid })
            }
        }
    }
    pub struct Commit<'repo> {
        // 这个指针总是指向一个可用的`git_commit`结构体。
        raw: *mut raw::git_commit,
        _marker: std::marker::PhantomData<&'repo Repository>,
    }
    impl Repository {
        pub fn find_commit(&self, oid: &Oid) -> Result<Commit> {
            let mut commit = ptr::null_mut();
            unsafe {
                check(raw::git_commit_lookup(&mut commit, self.raw, &oid.raw))?;
            }
            Ok(Commit {
                raw: commit,
                _marker: std::marker::PhantomData,
            })
        }
    }
    impl<'repo> Drop for Commit<'repo> {
        fn drop(&mut self) {
            unsafe {
                raw::git_commit_free(self.raw);
            }
        }
    }
    unsafe fn char_ptr_to_str<T>(_owner: &T, ptr: *const std::os::raw::c_char) -> Option<&str> {
        if ptr.is_null() {
            return None;
        } else {
            CStr::from_ptr(ptr).to_str().ok()
        }
    }
    impl<'repo> Commit<'repo> {
        pub fn author(&self) -> Signature {
            unsafe {
                Signature {
                    raw: raw::git_commit_author(self.raw),
                    _marker: std::marker::PhantomData,
                }
            }
        }
        pub fn message(&self) -> Option<&str> {
            unsafe {
                let message = raw::git_commit_message(self.raw);
                char_ptr_to_str(self, message)
            }
        }
    }
    pub struct Signature<'text> {
        raw: *const raw::git_signature,
        _marker: std::marker::PhantomData<&'text str>,
    }
    impl<'text> Signature<'text> {
        /// 以`&str`返回 author 的 name，
        /// 如果不是有效的 UTF-8 则返回`None`
        pub fn name(&self) -> Option<&str> {
            unsafe { char_ptr_to_str(self, (*self.raw).name) }
        }
        /// 以`&str`返回 author 的 email，
        /// 如果不是有效的 UTF-8 则返回`None`
        pub fn email(&self) -> Option<&str> {
            unsafe { char_ptr_to_str(self, (*self.raw).email) }
        }
    }
}
