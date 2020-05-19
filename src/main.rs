use nvml_wrapper::NVML;
use std::thread;
use std::sync::Arc;
use std::sync::atomic;
use subprocess;


fn main() {
    println!("GPULoad monitoring");
    let mut process = subprocess::Exec::shell("find / -type f > /dev/null 2> /dev/null").popen().unwrap();
    let child="firefox";
    let pid = process.pid();

    let started2 = Arc::new(atomic::AtomicBool::new(false));
    let finished2 = Arc::new(atomic::AtomicBool::new(false));

    let started = Arc::clone(&started2);
    let finished= Arc::clone(&finished2);

    let mut t = thread::spawn(move || {
        let nvml = NVML::init().unwrap();
        let dc = nvml.device_count().unwrap();
        println!("{} gpus found",dc);


        for gpu_id in (0..dc) {

            let device = nvml.device_by_index(gpu_id).unwrap();
            let memory_info = device.memory_info().unwrap();
            println!("{:?}",memory_info);

            let acc: usize = 0;



            while !finished.load(atomic::Ordering::Relaxed) {
                let processes = device.running_graphics_processes().unwrap();

                let mut found = false;
                for p in processes {
                    let name = nvml.sys_process_name(p.pid,65536).unwrap_or("".to_string()); // 65536 : max length for process name, otherwise truncated
                    if name.contains(child) {
                        let already_started = started.swap(true,atomic::Ordering::Relaxed);
                        if already_started == false {
                        println!("GPULoad started");
                        }
                    }
                    if started.load(atomic::Ordering::Relaxed) {
                        if name.contains(child) {
                            found = true;
                            println!("{:?}",p);
                        }
                    }

                }
                    if found==false {
                        finished.swap(true,atomic::Ordering::Relaxed);
                    }

            }
            println!("GPULoad finished");
        }

    });
    process.wait();
    finished2.swap(true,atomic::Ordering::Relaxed);

    t.join();


}
