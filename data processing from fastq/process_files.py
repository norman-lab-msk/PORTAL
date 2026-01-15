import gzip
import numpy as np

trans = str.maketrans('ATGC', 'TACG')

def read_fastq(fn):
    n = 4
    #sequence = {}
    quality = []
    sequence = []
    
    with gzip.open(fn, 'rt') as fh:
        lines = []
        for line in fh:
            lines.append(line.rstrip())
            if len(lines) == n:
                name = lines[0].split(" ")[0]
                if name[0] != "@":
                    raise ValueError(
                        "Records in Fastq files should start with '@' character"
                    )
                #sequence[name[1:]] = lines[1]
                sequence.append(lines[1])
                quality.append(lines[3])
                lines = []
    return np.array(sequence), np.array(quality)

def read_fastq_umi(fn):
    n = 4
    #sequence = {}
    #quality = {}
    sequence = []
    
    with gzip.open(fn, 'rt') as fh:
        lines = []
        for line in fh:
            lines.append(line.rstrip())
            if len(lines) == n:
                name = lines[0].split(" ")[0]
                if name[0] != "@":
                    raise ValueError(
                        "Records in Fastq files should start with '@' character"
                    )
                #sequence[name[1:]] = lines[1]
                sequence.append(lines[1][8:])
                #quality[name[1:]] = lines[3]
                lines = []
    return np.array(sequence)#, quality


def read_fastq_g2(fn):
    n = 4
    sequence = []
    
    with gzip.open(fn, 'rt') as fh:
        lines = []
        for line in fh:
            lines.append(line.rstrip())
            if len(lines) == n:
                name = lines[0].split(" ")[0]
                if name[0] != "@":
                    raise ValueError(
                        "Records in Fastq files should start with '@' character"
                    )
                try:
                    seq = lines[1][140:]
                    index = seq.index("CAAAC")
                    sequence.append(seq[index+5:index+25].translate(trans)[::-1])
                except Exception:
                    try:
                        seq = lines[1][140:]
                        index = seq.index("AAAC")
                        sequence.append(seq[index+4:index+24].translate(trans)[::-1])
                    except:
                        sequence.append(seq[-23:-3].translate(trans)[::-1])
                lines = []
    return np.array(sequence)


def read_fastq_b(fn):
    n = 4
    sequence = []
    
    with gzip.open(fn, 'rt') as fh:
        lines = []
        for line in fh:
            lines.append(line.rstrip())
            if len(lines) == n:
                name = lines[0].split(" ")[0]
                if name[0] != "@":
                    raise ValueError(
                        "Records in Fastq files should start with '@' character"
                    )
                sequence.append(lines[1][:25].translate(trans)[::-1])
                lines = []
    return np.array(sequence)


def read_fastq_g1(fn):
    n = 4
    sequence = []
    
    with gzip.open(fn, 'rt') as fh:
        lines = []
        for line in fh:
            lines.append(line.rstrip())
            if len(lines) == n:
                name = lines[0].split(" ")[0]
                if name[0] != "@":
                    raise ValueError(
                        "Records in Fastq files should start with '@' character"
                    )
                try:
                    index = lines[1].index("TAAAC")
                    sequence.append(lines[1][index+5:index+25].translate(trans)[::-1])
                except Exception:
                    try:
                        index = lines[1].index("AAAC")
                        sequence.append(lines[1][index+4:index+24].translate(trans)[::-1])
                    except:
                        sequence.append("")
                lines = []
    return np.array(sequence)


def read_fastq_g(fn):
    n = 4
    sequence = []
    
    with gzip.open(fn, 'rt') as fh:
        lines = []
        for line in fh:
            lines.append(line.rstrip())
            if len(lines) == n:
                name = lines[0].split(" ")[0]
                if name[0] != "@":
                    raise ValueError(
                        "Records in Fastq files should start with '@' character"
                    )
                try:
                    index = lines[1].index("TGTTG")
                    sequence.append(lines[1][index+5:index+25].translate(trans)[::-1])
                except Exception:
                    try:
                        index = lines[1].index("GTTG")
                        sequence.append(lines[1][index+4:index+24].translate(trans)[::-1])
                    except:
                        sequence.append("")
                lines = []
    return np.array(sequence)
