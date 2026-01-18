#!/bin/bash

# Sniffles2 SV Analysis
# Usage: ./sniffles_analysis.sh <input.bam> <reference.fasta> <output.txt>

if [ $# -ne 3 ]; then
    echo "Usage: $0 <input.bam> <reference.fasta> <output.txt>"
    exit 1
fi

BAM_FILE=$1
REF_FASTA=$2
OUTPUT_FILE=$3

# Check files and index BAM if needed
if [ ! -f "$BAM_FILE" ] || [ ! -f "$REF_FASTA" ]; then
    echo "Error: Input files not found!"
    exit 1
fi

if [ ! -f "${BAM_FILE}.bai" ]; then
    samtools index "$BAM_FILE"
fi

echo "Running Sniffles2..."

# Run Sniffles2
VCF_FILE="${OUTPUT_FILE%.txt}.vcf"
sniffles --input "$BAM_FILE" --vcf "$VCF_FILE" --reference "$REF_FASTA" \
  --minsvlen 10 \
  --minsupport 1 \
  --mapq 0 \
  --mosaic \
  --no-qc \
  --no-consensus \
  --qc-coverage 0 \
  --long-del-coverage 0.0 \
  --long-dup-coverage 100.0 \
  --mosaic-qc-coverage-max-change-frac 1.0 \
  --qc-coverage-max-change-frac 1.0 \
  --long-inv-length 1 \
  --max-splits-kb 100.0 \
  --max-splits-base 1000 \
  --min-alignment-length 50 \
  --cluster-merge-pos 10 \
  --cluster-merge-len 0.05 \
  --output-rnames \
  --threads 8 \
  --allow-overwrite > /dev/null 2>&1

if [ ! -f "$VCF_FILE" ]; then
    echo "Error: Sniffles2 failed!"
    exit 1
fi

# Get total reads in BAM
TOTAL_READS=$(samtools view -c -F 256 -F 2048 "$BAM_FILE")

# Analyze results
TOTAL_SVS=$(grep -v '^#' "$VCF_FILE" | wc -l)
DELETIONS=$(grep -v '^#' "$VCF_FILE" | grep 'SVTYPE=DEL' | wc -l)
INSERTIONS=$(grep -v '^#' "$VCF_FILE" | grep 'SVTYPE=INS' | wc -l)
INVERSIONS=$(grep -v '^#' "$VCF_FILE" | grep 'SVTYPE=INV' | wc -l)
DUPLICATIONS=$(grep -v '^#' "$VCF_FILE" | grep 'SVTYPE=DUP' | wc -l)
TRANSLOCATIONS=$(grep -v '^#' "$VCF_FILE" | grep 'SVTYPE=BND' | wc -l)

# Calculate total supporting reads and percentage
SUPPORT_ANALYSIS=$(grep -v '^#' "$VCF_FILE" | awk -F'\t' '{
    info = $8
    if (match(info, /SUPPORT=([0-9]+)/, arr)) {
        support = arr[1]
        total_support += support
        if (support == 1) single_support++
        else if (support <= 5) low_support++
        else high_support++
        total_svs++
    }
}
END {
    print total_support+0, single_support+0, low_support+0, high_support+0, total_svs+0
}')

SUPPORTING_READS=$(echo "$SUPPORT_ANALYSIS" | cut -d' ' -f1)
SINGLE_SUPPORT=$(echo "$SUPPORT_ANALYSIS" | cut -d' ' -f2)
LOW_SUPPORT=$(echo "$SUPPORT_ANALYSIS" | cut -d' ' -f3)
HIGH_SUPPORT=$(echo "$SUPPORT_ANALYSIS" | cut -d' ' -f4)

SV_PERCENTAGE=$(echo "scale=2; $SUPPORTING_READS * 100 / $TOTAL_READS" | bc -l 2>/dev/null || echo "0")

# Extract read names supporting SVs for filtering
grep -v '^#' "$VCF_FILE" | awk -F'\t' '{
    info = $8
    if (match(info, /RNAMES=([^;]+)/, arr)) {
        split(arr[1], reads, ",")
        for (i in reads) print reads[i]
    }
}' | sort -u > "${OUTPUT_FILE%.txt}_sv_reads.txt"

SV_READ_COUNT=$(wc -l < "${OUTPUT_FILE%.txt}_sv_reads.txt")

# Create filtered BAM without SV reads
echo "Creating filtered BAM without SV reads..."
samtools view -h -F 256 -F 2048 "$BAM_FILE" | \
awk 'BEGIN{while((getline < "'"${OUTPUT_FILE%.txt}"'_sv_reads.txt") > 0) exclude[$0]=1} 
     /^@/ {print} 
     !/^@/ {if(!($1 in exclude)) print}' | \
samtools view -bS - > "${OUTPUT_FILE%.txt}_no_sv.bam"

samtools index "${OUTPUT_FILE%.txt}_no_sv.bam"

# Output results
cat > "$OUTPUT_FILE" << EOF
Sensitive Sniffles2 Analysis Results
====================================

BASIC STATISTICS
================
Total structural variants: $TOTAL_SVS
- Deletions: $DELETIONS
- Insertions: $INSERTIONS
- Inversions: $INVERSIONS
- Duplications: $DUPLICATIONS
- Translocations/Breakends: $TRANSLOCATIONS

SUPPORT ANALYSIS
================
Single-read support (1 read): $SINGLE_SUPPORT SVs
Low support (2-5 reads): $LOW_SUPPORT SVs
High support (>5 reads): $HIGH_SUPPORT SVs

READ STATISTICS
===============
Total reads: $TOTAL_READS
Unique reads supporting SVs: $SV_READ_COUNT
Total SV read instances: $SUPPORTING_READS
Percentage with SVs: $SV_PERCENTAGE%

FILES CREATED
=============
VCF file: $VCF_FILE
SV read names: ${OUTPUT_FILE%.txt}_sv_reads.txt
Filtered BAM (no SV reads): ${OUTPUT_FILE%.txt}_no_sv.bam
EOF

echo "Results saved to: $OUTPUT_FILE"
echo "Total SVs: $TOTAL_SVS"
echo "Reads with SVs: $SUPPORTING_READS ($SV_PERCENTAGE%)"
