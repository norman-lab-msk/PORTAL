#!/bin/bash

# FASTQ to BAM
# Usage: ./fastq_to_bam.sh <reads.fastq.gz> <masked_reference.fasta> <output_prefix>

if [ $# -ne 3 ]; then
    echo "Usage: $0 <reads.fastq.gz> <masked_reference.fasta> <output_prefix>"
    echo "Example: $0 hifi_reads.fastq.gz masked_ref.fasta sample1"
    exit 1
fi

FASTQ=$1
REF=$2
PREFIX=$3

echo "Processing: $FASTQ"
echo "Reference: $REF"
echo "Output prefix: $PREFIX"

# Step 1: Align with minimap2
echo "Step 1: Aligning reads with minimap2..."
minimap2 -ax map-hifi "$REF" "$FASTQ" > "${PREFIX}_aligned.sam"

# Step 2: Convert to BAM
echo "Step 2: Converting to BAM..."
samtools view -bS "${PREFIX}_aligned.sam" > "${PREFIX}_aligned.bam"

# Step 3: Sort BAM
echo "Step 3: Sorting BAM..."
samtools sort "${PREFIX}_aligned.bam" -o "${PREFIX}_sorted.bam"

# Step 4: Index BAM
echo "Step 4: Indexing BAM..."
samtools index "${PREFIX}_sorted.bam"

# Clean up intermediate files
rm -f "${PREFIX}_aligned.sam" "${PREFIX}_aligned.bam"

echo ""
echo "Pipeline complete!"
echo "Output: ${PREFIX}_sorted.bam"
echo ""
echo "Ready for Sniffles analysis:"
echo "./sniffles_analysis.sh ${PREFIX}_sorted.bam $REF ${PREFIX}_sniffles_results.txt"