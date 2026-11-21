#pragma once
// SPDX-License-Identifier: MIT

#include "aiter_tensor.h"

#include <string_view>

namespace aiter {

namespace detail {

constexpr bool q4_group64_is_tuned_rx_9070_xt(int pci_chip_id,
                                              int multiprocessor_count,
                                              std::string_view name) noexcept
{
    return pci_chip_id == 0x7550 && multiprocessor_count == 32 &&
           (name.empty() || name == "AMD Radeon RX 9070 XT");
}

static_assert(q4_group64_is_tuned_rx_9070_xt(0x7550, 32, "AMD Radeon RX 9070 XT"));
static_assert(q4_group64_is_tuned_rx_9070_xt(0x7550, 32, ""));
static_assert(!q4_group64_is_tuned_rx_9070_xt(0x7550, 28, "AMD Radeon RX 9070"));
static_assert(!q4_group64_is_tuned_rx_9070_xt(0x7550, 24, "AMD Radeon RX 9070 GRE"));
static_assert(!q4_group64_is_tuned_rx_9070_xt(0x7551, 32, "AMD Radeon AI PRO R9700"));
static_assert(!q4_group64_is_tuned_rx_9070_xt(0x7550, 32, "AMD Radeon AI PRO R9700"));
static_assert(!q4_group64_is_tuned_rx_9070_xt(0x7550, 32, "AMD Radeon RX 9070 XT Future"));
static_assert(!q4_group64_is_tuned_rx_9070_xt(0, 0, "unknown"));

} // namespace detail

void q4_group64_gemv_out(aiter_tensor_t& x,
                         aiter_tensor_t& packed_weight,
                         aiter_tensor_t& out,
                         int mapping);

} // namespace aiter
