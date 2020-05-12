import ftplib
import os
import subprocess

from tqdm import tqdm


def download_pmc_bioc():
    os.makedirs("../data/pmc_bioc", exist_ok=True)
    print("Downloading PMC BioC...")
    processes = []
    try:
        ftp = ftplib.FTP("ftp.ncbi.nlm.nih.gov")
        ftp.login("anonymous", "foo")
        ftp.cwd("pub/wilbur/BioC-PMC/")
        files = [f for f in ftp.mlsd(".") if "json" in f[0] and "tar.gz" in f[0] and "ascii" in f[0]]
        for fname, metadata in tqdm(list(files)):

            if (os.path.isfile(fname)
                and os.path.getsize(fname) == int(metadata['size'])):
                print(fname + " already downloaded. Skipping...")
                continue
            
            with open(os.path.join("pmc_bioc", fname), 'wb') as f:
                ftp.retrbinary('RETR %s' % fname, f.write)
            processes.append(subprocess.Popen(['tar', 'xf', os.path.join('pmc_bioc', fname)]))
    finally:
        ftp.close()

    for process in processes:
        process.wait()
    

if __name__ == '__main__':
    download_pmc_bioc()

        

        
