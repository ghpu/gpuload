use nvml_wrapper::NVML;
use procinfo::pid;
use std::env;
use std::process;
use std::sync::atomic;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time;

fn get_parents(pid: u32) -> Vec<u32> {
    let stat = pid::stat(pid as i32);
    match stat {
        Ok(s) => match s.ppid {
            0 => vec![],
            1 => vec![],
            p => {
                if p == pid as i32 {
                    vec![]
                } else {
                    let mut result = vec![s.ppid as u32];
                    result.extend_from_slice(&get_parents(s.ppid as u32));
                    result
                }
            }
        },
        _ => vec![],
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    eprintln!("GPULoad monitoring, revision {}", env!("VERGEN_SHA_SHORT"));
    if args.len() < 2 {
        eprintln!("Syntax : {} child_process child_args...", args[0]);
        std::process::exit(1);
    }

    let nvml = NVML::init().unwrap();
    let dc = nvml.device_count().unwrap();
    eprintln!("{} gpus found", dc);
    let pid = process::id();

    let stats2 = Arc::new(Mutex::new(vec![0.0; 2 * dc as usize]));
    let nbsamples2 = Arc::new(Mutex::new(vec![0.0; 2 * dc as usize]));

    let mut process = subprocess::Exec::cmd(args[1].clone())
        .args(&args[2..])
        .popen()
        .unwrap();
    let process_id = process.pid().unwrap();

    let started2 = Arc::new(atomic::AtomicBool::new(false));
    let finished2 = Arc::new(atomic::AtomicBool::new(false));

    let started = Arc::clone(&started2);
    let finished = Arc::clone(&finished2);
    let stats = Arc::clone(&stats2);
    let nbsamples = Arc::clone(&nbsamples2);

    let t = thread::spawn(move || {
        while !finished.load(atomic::Ordering::Relaxed) {
            let mut found = false;
            for gpu_id in 0..dc {
                let device = nvml.device_by_index(gpu_id).unwrap();

                let mut acc_mem_used: u64 = 0;
                let processes = device.running_compute_processes().unwrap();
                let urate = device.utilization_rates().unwrap();
                let mut old = stats.lock().unwrap();
                let mut nbs = nbsamples.lock().unwrap();

                for p in processes {
                    let parents = get_parents(p.pid);

                    if parents.contains(&pid) {
                        let already_started = started.swap(true, atomic::Ordering::Relaxed);
                        if !already_started {
                            eprintln!("GPULoad started");
                        }
                    }
                    if started.load(atomic::Ordering::Relaxed) && parents.contains(&pid) {
                        found = true;
                        acc_mem_used += match p.used_gpu_memory {
                            nvml_wrapper::enums::device::UsedGpuMemory::Used(t) => t,
                            _ => 0,
                        };
                        old[gpu_id as usize] += urate.gpu as f32;
                        nbs[gpu_id as usize] += 1.0;
                    }
                }
                old[dc as usize + gpu_id as usize] += acc_mem_used as f32;
                drop(old);
            }

            if started.load(atomic::Ordering::Relaxed) && !found {
                finished.swap(true, atomic::Ordering::Relaxed);
            } else {
                thread::sleep(time::Duration::from_millis(1000));
            }
        }
        eprintln!("GPULoad finished");
    });
    ctrlc::set_handler(move || {
        unsafe { libc::kill(process_id as i32, libc::SIGINT) };
    })
    .expect("Cannot set SIGINT handler");
    process.wait().unwrap();
    finished2.swap(true, atomic::Ordering::Relaxed);

    t.join().unwrap();
    for gpu_id in 0..dc {
        let s = stats2.lock().unwrap();
        let nbs = nbsamples2.lock().unwrap();
        let mut nbs = nbs[gpu_id as usize];
        if nbs == 0.0 {
            nbs = 1.0
        };
        eprintln!(
            "GPULoad   gpu {}  kernel time use {:.2} %  memory used {:.0} bytes",
            gpu_id,
            s[gpu_id as usize] / nbs,
            s[gpu_id as usize + dc as usize] / nbs
        );
    }
}
