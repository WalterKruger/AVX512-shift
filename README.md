# Variable 8-bit shifts
Despite how well bit shift operations are support with AVX-512, it still lacks any 8-bit granular shifts. This repository aims to fill that gap by providing the most efficient implementations for all shift types, in which the amount being shifted is specified by the corresponding element. These implementations make heavy use of the many powerful AVX-512 instructions for the best performance.

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
TODO: Used as a 16-bit rotate (otherwise unsupported)
