# Variable 8-bit shifts
Despite how well bit shift operations are support with AVX-512, it still lacks any 8-bit granular shifts. This repository aims to fill that gap by providing the most efficient implementations for all shift types, in which the amount being shifted is specified by the corresponding element. These implementations make heavy use of the many powerful AVX-512 instructions for the best performance.

This repository was inspired by the [Anime Tosho’s “out-of-band” galois-field-affine uses](https://gist.github.com/animetosho/6cb732ccb5ecd86675ca0a442b3c0622#byte-wise-variable-shift). A few of his variable shift implementations are included. Also, check out [Wunkolo’s gf2p8affine-based 8-bit shifts](https://wunkolo.github.io/post/2020/11/gf2p8affineqb-int8-shifting/) if you need to shift all values by the same amount.

# Amounts greater than the element size
Each implementation has different behaviors when the shift amount is greater or equal to the element size. Some algorithms require a specific overflow behavior, so it may be beneficial to choose the implementation based on that. Below is a list of behaviors exhibited when that occur:

- **Modular**: The shift amount is modulo the element size. E.g. `x << 12 = x << 4`
- **Saturating**: The source bits can be completely shifted away. Logical shifts zero the result, arithmetic shift completely fill the output with the sign.
- **Garbage**: The result is set to an unusual or non-sensical value.

Note that some implementations have multiple behaviours at once, like saturating mod 16.

# Implementation comparison
All tests were performed on my `Ryzen 5 8400f`, compiled using `GCC 15.1.0` @ `-O2 -march=native`.
The `perf` column measure how many seconds each method took perform 30 billion operations on random data using a loop. Each function was presumably inlined and thus constants were pulled outside of said loop.

The instruction column don’t include loading any vector or mmask constants.


## Logical left
| Name  | Instructions | Worst chain | Constants | Overflow | Perf |
| - | :-: | :-: | :-: | - | - |
| **gfmul** | 4 | `permb` =><br>`pand` =><br>`gf2p8mulb` | 2 | Modular | 12.63 |
| **via16** | 6 | `pandn` =><br>`psllvw` =><br>`ternlog` | 1 | Saturation | 12.62 |

Either option performs the same, so choose based on overflow behaviour.

## Logical right
| Name  | Instructions | Worst chain | Constants | Overflow | Perf |
| - | :-: | :-: | :-: | - | - |
| **multishift** | 4 | `ternlog` =><br>`multishift` =><br>`pand` | 3 | Modular | 12.77 |
| **via16** | 6 | `pand` =><br>`psrlvw` =><br>`ternlog` | 1 | Saturation | 12.75 |
| **revLeft** | 6 | `gf2p8affine` =><br>`permb` =><br>`pand` =><br>`gf2p8mulb` =><br>`gf2p8affine` | 2 | Saturation | 13.63 |

Choose `multishift` for modular overflow or `via16` for saturation.

## Arithmetic right
| Name  | Instructions | Worst chain | Constants | Overflow | Perf |
| - | :-: | :-: | :-: | - | - |
| **multi** | 5 | `ternlog` =><br>`multishift` =><br>`ternlog` | 3 | Modular | 12.65 |
| **via16LUT** | 7 | `psrlw` =><br>`pshrdvw` =><br>`movdqu8{k}` =><br>`ternlog` | 1+mmask | Sat % 16 | 14.74 |
| **16SignExt** | 6 | `pmaddubsw` =><br>`psravw` =><br>`ternlog` | 2 | Saturation | 12.66 |
| **2multi** | 5 | `pmaddubsw` =><br>`multishift` =><br>`multishift{k}` | 3+mmask | Modular | 12.64 |

Choose `multishift` for modular overflow or `16SignExt` for saturation. I don’t recommend `2multi` as multishift with a mask has relatively low latency on Zen4.

## Rotation left
| Name  | Instructions | Worst chain | Constants | Overflow | Perf |
| - | :-: | :-: | :-: | - | - |
| **leftRight** | 7 | `permb` =><br>`pand` =><br>`gf2p8mulb` =><br>`ternlog` | 3 | Garbage | 14.77 |
| **bitByBit** | 6 | `ptestmb` =><br>`psubb{k}` =><br>`gfaffine{k}` =><br>`gfaffine{k}` | 5 | Modular | 16.87 |
| **via16** | 6 | `psufb` =><br>`pshldvw` =><br>`movdqu8{k}` | 2+mmask | Modular | 12.66 |

`via16` is recommended due to its performance.

## Rotation right
| Name  | Instructions | Worst chain | Constants | Overflow | Perf |
| - | :-: | :-: | :-: | - | - |
| **bitByBit** | 6 | `ptestmb` =><br>`gfaffine{k}` =><br>`gfaffine{k}` =><br>`gfaffine{k}` | 6 | Modular | 19.14 |
| **2multi** | 5 | `pshufb` =><br>`multishift` =><br>`multishift{k}` | 4+mmask | Modular | 12.72 |
| **via16** | 6 | `pshufb` =><br>`pshufb` =><br>`movdqu8{k}` | 2+mmask | Modular | 12.71 |

Either `2multi` or `via16`.


# Specific useful intrinsics
## `gf2p8mul`
Multiplying values by a power of two is equivalent to a bit shift. Unfortunately, there is no low multiplication intrinsic that perform an `8 * 8` => 8-bit like `_mm_mullo_epi16`. However, it does provide an 8-bit finite field multiplication which performs a full carryless multiplication then reduces it to 8-bits.

If we prevent any bits from ending up in the high part of the product, the reduction step doesn’t occur, and we effectively get a regular carryless multiplication. Since multiplying by a power of two can’t cause any carries, we can use it to shift each element by a different amount.

## Shuffles
The variable shuffle intrinsics can be used as a small LUT when we use the index as the input. This is useful for obtaining masks and transforming shift amounts into powers of two. Unlike `_mm_shuffle_epi8`, `_mm_permutexvar_epi8` doesn’t zero out the value when the index has its most significant bit set, making it more useful for this purpose.

In addition to LUTs, they are also useful for duplicating values to fill their 16-bit lane, which is used when using the 16-bit shifts.

## `maddubs`
This unusual intrinsic first sign extends the unsigned 8-bit values in the first operand and the signed values in the second. These are then multiplied producing a signed 16-bit product, which is then added to the adjacent produce and placed in each 16-bit lane.

This can be used to sign extend each even 8-bit value into the odd, useful for arithmetic shifts.

## `multishift`
Although no direct 8-bit shifts exist, this intrinsic behaves very similarly to one. For each 8-bit value, it takes the value in the adjacent 64-bit lane, performs a rotation right shift by that amount, and keeps the lower 8-bits.

If we offset the shift count by the offset in each lane and modulo by 8, we almost get an 8-bit shift. Since each lane’s offset is greater than 8 (or zero), a bitwise OR can perform the offset after the amount has been masked. We can thus use `_mm_ternarylogic_epi32` to perform both the masking and offsetting in a single instruction.

## Double Precision 16-bit shift
AVX-512 introduced “double precision” shifts, where each element is concatenated to twice its width using the adjacent element before being shifted, with only the lower 16-bits being kept. This can be used to emulate the missing 16-bit rotational shifts by using the same value in both the high and low parts.

Notably, these shifts are modular rather than saturating unlike all other non-rotation SIMD shifts. Obviously, this is helpful for implementing modular overflow, but by being modular it effectively ignores the upper 8-bit shift amount saving on a mask.

# `gf2p8affineqb`
Although this instruction was intended for cryptography, it also can perform arbitrary fixed byte permutations. It is used as a single instruction 8-bit rotations by a fixed amount, in conjunction with a merge mask.
