#[cfg(target_os = "windows")]
pub fn meminfo_main() {
    use winapi::shared::basetsd::SIZE_T;
    use winapi::shared::minwindef::LPVOID;
    use winapi::shared::minwindef::{BYTE, DWORD, FALSE, PDWORD, TRUE, ULONG, USHORT, WORD};
    use winapi::shared::ntdef::PVOID;
    use winapi::um::sysinfoapi::LPSYSTEM_INFO;
    use winapi::um::sysinfoapi::SYSTEM_INFO;
    use winapi::um::winnt::HANDLE;
    use winapi::um::winnt::MEMORY_BASIC_INFORMATION;

    use winapi::um::memoryapi::VirtualQueryEx;
    use winapi::um::processthreadsapi::GetCurrentProcess;
    use winapi::um::processthreadsapi::GetCurrentProcessId;
    use winapi::um::sysinfoapi::GetSystemInfo;

    let this_pid: DWORD;
    let this_proc: HANDLE;
    let min_addr: LPVOID;
    let max_addr: LPVOID;
    let mut base_addr: PVOID;
    let mut proc_info: SYSTEM_INFO;
    let mut mem_info: MEMORY_BASIC_INFORMATION;

    const MEMINFO_SIZE: usize = std::mem::size_of::<MEMORY_BASIC_INFORMATION>();

    unsafe {
        base_addr = std::mem::zeroed();
        proc_info = std::mem::zeroed();
        mem_info = std::mem::zeroed();
    }

    unsafe {
        this_pid = GetCurrentProcessId();
        this_proc = GetCurrentProcess();
        GetSystemInfo(&mut proc_info as LPSYSTEM_INFO); // <10>
    };

    min_addr = proc_info.lpMinimumApplicationAddress;
    max_addr = proc_info.lpMaximumApplicationAddress;

    println!("{:?} @ {:p}", this_pid, this_proc);
    println!(
        "processor: {}, pagesize: {}",
        proc_info.dwNumberOfProcessors, proc_info.dwPageSize
    );
    println!("min: {:p}, max: {:p}", min_addr, max_addr);

    loop {
        let rc: SIZE_T =
            unsafe { VirtualQueryEx(this_proc, base_addr, &mut mem_info, MEMINFO_SIZE as SIZE_T) };

        if rc == 0 {
            break;
        }

        println!(
            "BaseAddress: {} RegionSize: {}",
            mem_info.BaseAddress as u64, mem_info.RegionSize as u64
        );
        base_addr = ((base_addr as u64) + mem_info.RegionSize as u64) as PVOID;
    }
}
