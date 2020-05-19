use nvml_wrapper::NVML;
use std::env;
use std::process;
use std::thread;
use std::time;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic;
use subprocess;


fn main() {
    eprintln!("GPULoad monitoring");
    let args: Vec<String>=env::args().collect();
    eprintln!("GPULoad revision {}",env!("VERGEN_SHA_SHORT"));
    if args.len() < 2 {
        eprintln!("Syntax : {} child_process child_args...", args[0]);
        std::process::exit(1);
    }

    let nvml = NVML::init().unwrap();
    let dc = nvml.device_count().unwrap();
    eprintln!("{} gpus found",dc);

    let stats2 = Arc::new(Mutex::new(vec![0.0;2* dc as usize]));
    let nbsamples2 = Arc::new(Mutex::new(0.0));


    let exec_path = args[1].clone();
    let child :String = match exec_path.rsplit("/").next() {
        Some(p) => p.into(),
        None => exec_path.into()
    };
    

    let mut process = subprocess::Exec::cmd(args[1].clone()).args(&args[2..]).popen().unwrap();
    let pid = process.pid();

    let started2 = Arc::new(atomic::AtomicBool::new(false));
    let finished2 = Arc::new(atomic::AtomicBool::new(false));

    let started = Arc::clone(&started2);
    let finished= Arc::clone(&finished2);
    let stats = Arc::clone(&stats2);
    let nbsamples = Arc::clone(&nbsamples2);

    let mut t = thread::spawn(move || {


        for gpu_id in 0..dc {

            let device = nvml.device_by_index(gpu_id).unwrap();
            let memory_info = device.memory_info().unwrap();
            eprintln!("{:?}",memory_info);

            while !finished.load(atomic::Ordering::Relaxed) {
                let mut acc_mem_used: u64 = 0;
                let processes = device.running_compute_processes().unwrap();
                let urate = device.utilization_rates().unwrap();
                let mut old = stats.lock().unwrap();

                let mut found = false;
                for p in processes {
                    let name = nvml.sys_process_name(p.pid,65536).unwrap_or("".to_string()); // 65536 : max length for process name, otherwise truncated
                    if name.contains(&child) {
                        let already_started = started.swap(true,atomic::Ordering::Relaxed);
                        if already_started == false {
                            eprintln!("GPULoad started");
                        }
                    }
                    if started.load(atomic::Ordering::Relaxed) {
                        if name.contains(&child) {
                            found = true;
                            acc_mem_used += match p.used_gpu_memory { nvml_wrapper::enums::device::UsedGpuMemory::Used(t) => t, _ => 0};
                            old[gpu_id as usize] += urate.gpu as f32;
                            *nbsamples.lock().unwrap() += 1.0;
                        }
                    }

                }
                eprintln!("Used memory : {}",acc_mem_used);
                eprintln!("Used gpu {}: kernel {} % , memory {} %",gpu_id, urate.gpu, urate.memory);

                old[dc as usize + gpu_id as usize] += acc_mem_used as f32;
                drop(old);


                if started.load(atomic::Ordering::Relaxed) && found==false {
                    finished.swap(true,atomic::Ordering::Relaxed);
                } else {
                    thread::sleep(time::Duration::from_millis(1000));
                }



            }
            eprintln!("GPULoad finished");
        }

    });
    process.wait();
    finished2.swap(true,atomic::Ordering::Relaxed);

    t.join();
    for gpu_id in 0..dc {
        let s = stats2.lock().unwrap();
        let mut nbs = nbsamples2.lock().unwrap();
        if *nbs == 0.0 { *nbs = 1.0};
        println!("GPULoad   gpu {}  kernel time use {:.2} %  memory used {:.0} bytes", gpu_id, s[gpu_id as usize] / *nbs, s[gpu_id as usize + dc as usize] / *nbs);

    }



}
