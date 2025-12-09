#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Unified nyancat data generator with configurable compression modes.

Downloads animation data from klange/nyancat repository and applies either:
- Opcode-based RLE compression (baseline, 87% reduction)
- Delta frame encoding (advanced, 91% reduction)

Opcode format (baseline RLE):
  0x0X = SetColor (current color = X, 0-13)
  0x2Y = Repeat Y+1 times (1-16 pixels)
  0x3Y = Repeat (Y+1)*16 times (16-256 pixels)
  0xFF = EndOfFrame

Delta encoding format (--delta mode):
  Frame 0 (baseline):
    0x0X = SetColor (X = color 0-13)
    0x2Y = Repeat (Y+1) times (1-16 pixels)
    0x3Y = Repeat (Y+1)*16 times (16-256 pixels)
    0xFF = EndOfFrame

  Frame 1-11 (delta):
    0x0X = SetColor (X = color 0-13)
    0x1Y = Skip (Y+1) unchanged pixels (1-16)
    0x2Y = Repeat (Y+1) changed pixels (1-16)
    0x3Y = Skip (Y+1)*16 unchanged pixels (16-256)
    0x4Y = Repeat (Y+1)*16 changed pixels (16-256)
    0x5Y = Skip (Y+1)*64 unchanged pixels (64-1024)
    0xFF = EndOfFrame
"""

import argparse
import re
import sys
import urllib.request
from pathlib import Path
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import heapq

class HuffmanNode:
    def __init__(self, char: Optional[int], freq: int):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data: List[int]) -> HuffmanNode:
    # Frequency count
    freq = {}
    for char in data:
        freq[char] = freq.get(char, 0) + 1
    
    # Priority queue
    heap = [HuffmanNode(char, count) for char, count in freq.items()]
    heapq.heapify(heap)
    
    # Build tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
        
    return heap[0] if heap else None

def generate_codes(root: HuffmanNode, current_code: str, codes: Dict[int, Tuple[int, int]]):
    if root is None:
        return

    if root.char is not None:
        # Found leaf, store (int_value_of_code, length)
        codes[root.char] = (int(current_code, 2) if current_code else 0, len(current_code))
        return

    generate_codes(root.left, current_code + "0", codes)
    generate_codes(root.right, current_code + "1", codes)

def serialize_huffman_tree(root: HuffmanNode) -> Tuple[List[int], List[int]]:
    # Serialize Huffman Tree to flattened C-compatible arrays
    # Protocol:
    #   - Indices 0..255 are implicit Leaf Nodes (value = Opcode)
    #   - Indices 256..N are Internal Nodes
    #   - Array entries: value < 256 (Leaf), value >= 256 (Next Internal Node Index)
    
    # Map internal nodes to indices starting from 256
    # BFS traversal ensures parent nodes are processed before children regarding index assignment logic
    
    table_left = []
    table_right = []
    
    # Internal node queue for BFS construction
    # Note: Root is effectively Internal Node 0 (Index 256 in theoretical space, but 0 in array)
    internal_queue = [root]
    
    i = 0
    while i < len(internal_queue):
        curr = internal_queue[i] # This is internal node 'i'
        i += 1
        
        # Process Left Child
        if curr.left.char is not None:
            left_val = curr.left.char # Leaf
        else:
            left_val = 256 + len(internal_queue) # Next available internal index
            internal_queue.append(curr.left)
            
        # Process Right Child
        if curr.right.char is not None:
            right_val = curr.right.char # Leaf
        else:
            right_val = 256 + len(internal_queue)
            internal_queue.append(curr.right)
            
        table_left.append(left_val)
        table_right.append(right_val)
        
    return table_left, table_right

def compress_data_huffman(data: List[int], codes: Dict[int, Tuple[int, int]]) -> bytes:
    bit_buffer = 0
    bit_count = 0
    output = bytearray()
    
    for char in data:
        code, length = codes[char]
        
        # Pack bits LSB-first into byte stream (matches C decoder behavior)
        # Sequence: Bit 0 of Code -> LSB of Byte, Bit 1 -> Bit 1 ...
        for i in range(length):
            bit = (code >> (length - 1 - i)) & 1 # Extract bit from code (MSB to LSB as per string)
            bit_buffer |= (bit << bit_count)
            bit_count += 1
            if bit_count == 8:
                output.append(bit_buffer)
                bit_buffer = 0
                bit_count = 0
                
    if bit_count > 0:
        output.append(bit_buffer)
        
    return bytes(output)


def download_animation_data(url: str) -> str:
    """Download animation.c from GitHub repository."""
    try:
        with urllib.request.urlopen(url) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        print(f"Error downloading from {url}: {e}", file=sys.stderr)
        sys.exit(1)


def parse_animation_c(content: str) -> List[List[str]]:
    """
    Parse animation.c to extract frame data.

    Returns list of 12 frames, each frame is list of pixel strings.
    """
    frames = []

    # Find all frame arrays (frame0[] through frame11[])
    for frame_num in range(12):
        pattern = rf'const\s+char\s+\*\s*frame{frame_num}\[\]\s*=\s*\{{([^}}]+)\}}'
        match = re.search(pattern, content, re.DOTALL)

        if not match:
            print(f"Error: Could not find frame{frame_num}[] in animation.c", file=sys.stderr)
            sys.exit(1)

        frame_text = match.group(1)

        # Extract all quoted strings for this frame
        frame_lines = re.findall(r'"([^"]*)"', frame_text)

        # Concatenate all lines into single frame (64 lines × 64 chars = 4096 pixels)
        frame_data = ''.join(frame_lines)

        if len(frame_data) != 4096:
            print(f"Error: frame{frame_num} has {len(frame_data)} pixels, expected 4096", file=sys.stderr)
            sys.exit(1)

        frames.append(list(frame_data))

    return frames


def map_color_to_palette(char: str) -> int:
    """
    Map nyancat color character to palette index.

    Original mapping from klange/nyancat upstream:
    , = dark blue background
    . = white (stars)
    ' = black (border)
    @ = tan (poptart)
    $ = pink (poptart)
    - = red (poptart)
    > = red (rainbow)
    & = orange (rainbow)
    + = yellow (rainbow)
    # = green (rainbow)
    = = light blue (rainbow)
    ; = dark blue (rainbow)
    * = gray (cat face)
    % = pink (cheeks)
    """
    color_map = {
        ',': 0,   # Dark blue background
        '.': 1,   # White (stars)
        "'": 2,   # Black (border)
        '@': 3,   # Tan/Light pink (poptart) -> Light pink/beige
        '$': 5,   # Pink poptart -> Hot pink
        '-': 6,   # Red poptart
        '>': 6,   # Red rainbow (same as red poptart)
        '&': 7,   # Orange rainbow
        '+': 8,   # Yellow rainbow
        '#': 9,   # Green rainbow
        '=': 10,  # Light blue rainbow
        ';': 11,  # Dark blue/Purple rainbow -> Purple
        '*': 12,  # Gray cat face
        '%': 4,   # Pink cheeks
    }
    return color_map.get(char, 0)  # Default to background


def compress_frame_opcode_rle(pixels: List[str]) -> List[int]:
    """
    Compress frame using opcode-based RLE (baseline compression).

    Returns list of opcodes (integers 0-255).
    """
    if len(pixels) != 4096:
        print(f"Error: Frame must have 4096 pixels, got {len(pixels)}", file=sys.stderr)
        sys.exit(1)

    opcodes = []
    i = 0
    current_color = -1

    while i < len(pixels):
        color = map_color_to_palette(pixels[i])

        # Set color if different from current
        if color != current_color:
            opcodes.append(0x00 | color)  # SetColor opcode
            current_color = color

        # Count consecutive pixels of same color
        count = 1
        while i + count < len(pixels) and map_color_to_palette(pixels[i + count]) == color:
            count += 1

        # Encode run length with appropriate opcodes (may need multiple for long runs)
        remaining = count
        while remaining > 0:
            if remaining <= 16:
                # Short repeat: 0x2Y (1-16 pixels)
                opcodes.append(0x20 | (remaining - 1))
                remaining = 0
            elif remaining <= 256:
                # Long repeat: 0x3Y (16-256 pixels in multiples of 16)
                # Emit full chunks of 16
                chunks = min(remaining // 16, 16)  # Max 16 chunks = 256 pixels
                if chunks > 0:
                    opcodes.append(0x30 | (chunks - 1))
                    remaining -= chunks * 16
            else:
                # For very long runs (>256), emit max long repeat (256 pixels)
                opcodes.append(0x30 | 0x0F)  # 16 chunks = 256 pixels
                remaining -= 256

        i += count

    # End of frame marker
    opcodes.append(0xFF)

    return opcodes


def compress_delta_frame(prev_pixels: List[str], curr_pixels: List[str]) -> List[int]:
    """
    Compress delta frame using skip + repeat encoding.

    Returns list of opcodes exploiting temporal coherence.
    """
    if len(prev_pixels) != 4096 or len(curr_pixels) != 4096:
        print("Error: Frames must have 4096 pixels", file=sys.stderr)
        sys.exit(1)

    # Convert to color indices
    prev_colors = [map_color_to_palette(p) for p in prev_pixels]
    curr_colors = [map_color_to_palette(p) for p in curr_pixels]

    opcodes = []
    i = 0
    current_color = -1

    while i < 4096:
        # Count consecutive unchanged pixels
        skip_count = 0
        while i + skip_count < 4096 and prev_colors[i + skip_count] == curr_colors[i + skip_count]:
            skip_count += 1

        # Encode skip if any unchanged pixels
        if skip_count > 0:
            remaining_skip = skip_count
            while remaining_skip > 0:
                if remaining_skip <= 16:
                    # 0x1Y: Skip 1-16 unchanged pixels
                    opcodes.append(0x10 | (remaining_skip - 1))
                    remaining_skip = 0
                elif remaining_skip <= 256:
                    # 0x3Y: Skip 16-256 unchanged pixels (chunks of 16)
                    chunks = min(remaining_skip // 16, 16)
                    if chunks > 0:
                        opcodes.append(0x30 | (chunks - 1))
                        remaining_skip -= chunks * 16
                elif remaining_skip <= 1024:
                    # 0x5Y: Skip 64-1024 unchanged pixels (chunks of 64)
                    chunks = min(remaining_skip // 64, 16)
                    if chunks > 0:
                        opcodes.append(0x50 | (chunks - 1))
                        remaining_skip -= chunks * 64
                else:
                    # Max skip: 1024 pixels
                    opcodes.append(0x50 | 0x0F)
                    remaining_skip -= 1024

            i += skip_count
            if i >= 4096:
                break

        # Handle changed pixels
        color = curr_colors[i]
        if color != current_color:
            opcodes.append(0x00 | color)  # SetColor
            current_color = color

        # Count consecutive changed pixels of same color
        run_len = 1
        while i + run_len < 4096 and \
              curr_colors[i + run_len] == color and \
              prev_colors[i + run_len] != curr_colors[i + run_len]:
            run_len += 1

        # Encode changed run
        remaining_run = run_len
        while remaining_run > 0:
            if remaining_run <= 16:
                # 0x2Y: Repeat 1-16 changed pixels
                opcodes.append(0x20 | (remaining_run - 1))
                remaining_run = 0
            elif remaining_run <= 256:
                # 0x4Y: Repeat 16-256 changed pixels (chunks of 16)
                chunks = min(remaining_run // 16, 16)
                if chunks > 0:
                    opcodes.append(0x40 | (chunks - 1))
                    remaining_run -= chunks * 16
            else:
                # Max run: 256 pixels
                opcodes.append(0x40 | 0x0F)
                remaining_run -= 256

        i += run_len

    opcodes.append(0xFF)
    return opcodes


    opcodes.append(0xFF)
    return opcodes

def generate_header(frames: List[List[str]], output_path: Path, use_delta: bool = False, use_huffman: bool = False) -> None:
    """Generate nyancat-data.h with compressed frame data."""

    # Compress all frames (Opcode Generation Stage)
    raw_opcodes_per_frame = []
    
    if use_delta:
        # Frame 0: baseline RLE
        raw_opcodes_per_frame.append(compress_frame_opcode_rle(frames[0]))
        # Frames 1-11: delta encoding
        for i in range(1, 12):
            raw_opcodes_per_frame.append(compress_delta_frame(frames[i-1], frames[i]))
    else:
        # Baseline: opcode-RLE for all frames
        for frame in frames:
            raw_opcodes_per_frame.append(compress_frame_opcode_rle(frame))
            
    # Flatten all opcodes to build a single Global Huffman Tree (better compression than per-frame)
    all_opcodes = []
    for frame_ops in raw_opcodes_per_frame:
        all_opcodes.extend(frame_ops)
        
    final_data_bytes = []
    
    if use_huffman:
        print("Building Huffman Tree...")
        root = build_huffman_tree(all_opcodes)
        codes = {}
        generate_codes(root, "", codes)
        
        # Serialize Tree
        tree_left, tree_right = serialize_huffman_tree(root)
        
        # Compress Data
        # Compress frames individually to allow potential frame-based seeking/offsets
        # Though current Huffman implementation is global-stream based, we process frame-by-frame for logic clarity
        
        compressed_frames = []
        for ops in raw_opcodes_per_frame:
            compressed_frames.append(compress_data_huffman(ops, codes))
            
        # Flatten compressed bytes
        for cf in compressed_frames:
            final_data_bytes.extend(cf)
            
        # Calculate offsets (byte aligned)
        offsets = [0]
        for cf in compressed_frames[:-1]:
            offsets.append(offsets[-1] + len(cf))
            
    else:
        # Just use the raw opcodes (RLE/Delta) as the "compressed" stream
        # This matches original behavior
        final_data_bytes = all_opcodes
        
        # Recalculate offsets based on raw opcode counts
        offsets = [0]
        for ops in raw_opcodes_per_frame[:-1]:
            offsets.append(offsets[-1] + len(ops))

    total_original = 12 * 4096
    total_compressed = len(final_data_bytes)
    
    # Tree size in bytes (each internal node is 4 bytes: 2x uint16)
    tree_size = 0
    if use_huffman:
        tree_size = len(tree_left) * 4
        total_compressed += tree_size
        
    reduction = 100 - (total_compressed * 100 // total_original)

    print(f"\nTotal: {total_original} pixels → {total_compressed} bytes ({reduction}% reduction)")
    if use_huffman:
         print(f"  - Stream: {len(final_data_bytes)} bytes")
         print(f"  - Tree:   {tree_size} bytes ({len(tree_left)} nodes)")

    # Generate header file
    mode_str = "delta" if use_delta else "baseline"
    if use_huffman: mode_str += "+huffman"
    
    with open(output_path, 'w') as f:
        f.write(f"// SPDX-License-Identifier: MIT\n")
        f.write(f"// Auto-generated nyancat animation data with {mode_str} compression\n")
        f.write(f"// DO NOT EDIT - Generated by scripts/gen-nyancat-data.py\n\n")
        f.write(f"#ifndef NYANCAT_DATA_H\n#define NYANCAT_DATA_H\n\n")
        f.write(f"#include <stdint.h>\n\n")
        
        f.write(f"// Compression config\n")
        f.write(f"#define NYANCAT_MODE_DELTA {1 if use_delta else 0}\n")
        f.write(f"#define NYANCAT_MODE_HUFFMAN {1 if use_huffman else 0}\n\n")
        
        if use_huffman:
            f.write(f"// Huffman Tree ({len(tree_left)} internal nodes)\n")
            f.write(f"// Format: struct {{ uint16_t left; uint16_t right; }} nodes[];\n")
            f.write(f"// If value < 256, it is a leaf (opcode). If >= 256, it is index of child node (val - 256).\n")
            f.write(f"static const uint16_t nyancat_huffman_tree[{len(tree_left)}][2] = {{\n")
            for l, r in zip(tree_left, tree_right):
                f.write(f"    {{ {l}, {r} }},\n")
            f.write(f"}};\n\n")

        f.write(f"// Frame offset table\n")
        f.write(f"static const uint16_t nyancat_frame_offsets[12] = {{\n")
        for i in range(0, len(offsets), 6):
            chunk = offsets[i:i+6]
            f.write("    " + ", ".join(f"{os:5d}" for os in chunk))
            if i + 6 < len(offsets): f.write(",")
            f.write("\n")
        f.write("};\n\n")
        
        f.write(f"// Compressed data\n")
        f.write(f"static const uint8_t nyancat_compressed_data[{len(final_data_bytes)}] = {{\n")
        for i in range(0, len(final_data_bytes), 16):
            chunk = final_data_bytes[i:i+16]
            f.write("    " + ", ".join(f"0x{b:02x}" for b in chunk))
            if i + 16 < len(final_data_bytes): f.write(",")
            f.write("\n")
        f.write("};\n\n")
        f.write("#endif\n")

    print(f"\nGenerated: {output_path}")



def decompress_and_verify(frames: List[List[str]], use_delta: bool = False) -> bool:
    """
    Decompress compressed frames and verify against originals.

    Returns True if all frames match.
    """
    print("\n=== Verification Mode ===")

    all_match = True

    if use_delta:
        # Verify delta compression
        prev_frame = None
        for frame_idx, original_frame in enumerate(frames):
            if frame_idx == 0:
                # Baseline frame
                opcodes = compress_frame_opcode_rle(original_frame)
                decompressed = decompress_baseline(opcodes)
                prev_frame = original_frame
            else:
                # Delta frame
                opcodes = compress_delta_frame(prev_frame, original_frame)
                decompressed = decompress_delta(decompress_baseline(compress_frame_opcode_rle(prev_frame)), opcodes)
                prev_frame = original_frame

            # Verify
            original_colors = [map_color_to_palette(p) for p in original_frame]
            if len(decompressed) != 4096:
                print(f"Frame {frame_idx}: Length mismatch! Expected 4096, got {len(decompressed)}")
                all_match = False
            else:
                mismatches = sum(1 for a, b in zip(original_colors, decompressed) if a != b)
                if mismatches > 0:
                    print(f"Frame {frame_idx}: {mismatches} pixel mismatches")
                    all_match = False
                else:
                    print(f"Frame {frame_idx}: ✓ Perfect match ({len(opcodes)} opcodes)")
    else:
        # Verify baseline RLE
        for frame_idx, original_frame in enumerate(frames):
            opcodes = compress_frame_opcode_rle(original_frame)
            decompressed = decompress_baseline(opcodes)

            if len(decompressed) != 4096:
                print(f"Frame {frame_idx}: Length mismatch! Expected 4096, got {len(decompressed)}")
                all_match = False
                continue

            original_colors = [map_color_to_palette(p) for p in original_frame]
            mismatches = sum(1 for a, b in zip(original_colors, decompressed) if a != b)

            if mismatches > 0:
                print(f"Frame {frame_idx}: {mismatches} pixel mismatches")
                all_match = False
            else:
                print(f"Frame {frame_idx}: ✓ Perfect match ({len(opcodes)} opcodes)")

    return all_match


def decompress_baseline(opcodes: List[int]) -> List[int]:
    """Decompress baseline RLE opcodes to color indices."""
    decompressed = []
    current_color = 0
    i = 0

    while i < len(opcodes):
        opcode = opcodes[i]
        i += 1

        if opcode == 0xFF:
            break
        elif (opcode & 0xF0) == 0x00:
            current_color = opcode & 0x0F
        elif (opcode & 0xF0) == 0x20:
            count = (opcode & 0x0F) + 1
            decompressed.extend([current_color] * count)
        elif (opcode & 0xF0) == 0x30:
            count = ((opcode & 0x0F) + 1) * 16
            decompressed.extend([current_color] * count)

    return decompressed


def decompress_delta(prev_frame: List[int], opcodes: List[int]) -> List[int]:
    """Decompress delta frame opcodes using previous frame."""
    decompressed = list(prev_frame)  # Start with previous frame
    pos = 0
    current_color = 0
    i = 0

    while i < len(opcodes) and pos < 4096:
        opcode = opcodes[i]
        i += 1

        if opcode == 0xFF:
            break
        elif (opcode & 0xF0) == 0x00:
            current_color = opcode & 0x0F
        elif (opcode & 0xF0) == 0x10:
            pos += (opcode & 0x0F) + 1  # Skip unchanged
        elif (opcode & 0xF0) == 0x20:
            count = (opcode & 0x0F) + 1  # Repeat changed
            for _ in range(count):
                if pos < 4096:
                    decompressed[pos] = current_color
                    pos += 1
        elif (opcode & 0xF0) == 0x30:
            pos += ((opcode & 0x0F) + 1) * 16  # Skip unchanged (long)
        elif (opcode & 0xF0) == 0x40:
            count = ((opcode & 0x0F) + 1) * 16  # Repeat changed (long)
            for _ in range(count):
                if pos < 4096:
                    decompressed[pos] = current_color
                    pos += 1
        elif (opcode & 0xF0) == 0x50:
            pos += ((opcode & 0x0F) + 1) * 64  # Skip unchanged (very long)

    return decompressed


def main():
    parser = argparse.ArgumentParser(
        description="Generate nyancat-data.h with configurable compression"
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('nyancat-data.h'),
        help='Output header file path (default: nyancat-data.h)'
    )
    parser.add_argument(
        '--url',
        default='https://raw.githubusercontent.com/klange/nyancat/master/src/animation.c',
        help='URL to animation.c (default: klange/nyancat master)'
    )
    parser.add_argument(
        '--delta',
        action='store_true',
        help='Use delta frame compression (default: baseline RLE)'
    )
    parser.add_argument(
        '--huffman',
        action='store_true',
        help='Apply Huffman coding on top of RLE/Delta opcodes'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify compression/decompression without generating output'
    )

    args = parser.parse_args()

    # Download animation data
    print(f"Downloading from: {args.url}")
    content = download_animation_data(args.url)

    # Parse frames
    print("Parsing animation frames...")
    frames = parse_animation_c(content)
    print(f"Parsed {len(frames)} frames, {len(frames[0])} pixels each")

    # Verify mode
    if args.verify:
        success = decompress_and_verify(frames, args.delta)
        sys.exit(0 if success else 1)

    # Generate header
    compression_mode = "delta" if args.delta else "baseline"
    if args.huffman: compression_mode += "+huffman"
    
    print(f"\nCompressing frames with {compression_mode}...")
    generate_header(frames, args.output, args.delta, args.huffman)

    # Run verification
    print("\nVerifying compression...")
    if decompress_and_verify(frames, args.delta):
        print("\n✓ All frames verified successfully")
    else:
        print("\n✗ Verification failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
