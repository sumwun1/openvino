// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/cpu_kernel_executor_table.hpp"
#include <cpu/x64/brgemm/brgemm.hpp>

namespace ov {
namespace intel_cpu {
struct BrgemmKernelConfig : public snippets::KernelExecutorBase::GenericConfig {
public:
    BrgemmKernelConfig(const element::Type& in0_dtype, const element::Type& in1_dtype, float beta,
                       bool is_with_amx, bool is_with_comp,
                       size_t M = 0, size_t N = 0, size_t K = 0,
                       size_t LDA = 0, size_t LDB = 0, size_t LDC = 0);
    BrgemmKernelConfig() = default;
    bool is_completed() const override;
    size_t hash() const override { return m_hash; }
    std::shared_ptr<GenericConfig> clone() const override {
        return std::make_shared<BrgemmKernelConfig>(*this);
    }
    void update(dnnl_dim_t M, dnnl_dim_t N, dnnl_dim_t K, dnnl_dim_t LDA, dnnl_dim_t LDB, dnnl_dim_t LDC);

    dnnl_data_type_t get_dt_in0() const { return m_dt_in0; }
    dnnl_data_type_t get_dt_in1() const { return m_dt_in1; }

    dnnl::impl::cpu::x64::cpu_isa_t get_isa() const { return m_isa; }
    bool is_with_amx() const {return m_is_with_amx; }
    bool is_with_comp() const { return m_is_with_comp; }
    float get_beta() const { return m_beta; }

    dnnl_dim_t get_M() const { return m_M; }
    dnnl_dim_t get_N() const { return m_N; }
    dnnl_dim_t get_K() const { return m_K; }

    dnnl_dim_t get_LDA() const { return m_LDA; }
    dnnl_dim_t get_LDB() const { return m_LDB; }
    dnnl_dim_t get_LDC() const { return m_LDC; }

    explicit operator amx_tile_config_t() const;
    inline bool compatible(amx_tile_config_t* rhs) const {
        return rhs && rhs->M == m_M && rhs->N == m_N && rhs->K == m_K;
    }

#ifdef SNIPPETS_DEBUG_CAPS
    std::string to_string() const override;
#endif

private:
    size_t compute_hash() const;
    dnnl_data_type_t m_dt_in0 {dnnl_f32}, m_dt_in1 {dnnl_f32};
    bool m_is_with_amx {false};
    bool m_is_with_comp {false};
    float m_beta {0};
    dnnl::impl::cpu::x64::cpu_isa_t m_isa {dnnl::impl::cpu::x64::isa_undef};
    dnnl_dim_t m_M {0}, m_N {0}, m_K {0}, m_LDA {0}, m_LDB {0}, m_LDC {0};
    size_t m_hash {SIZE_MAX};
};

struct BrgemmCompiledKernel {
    std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> compiled_kernel = nullptr;
    // Note: Palette is treated as a part of a kernel because it is initialized during the kernel compilation stage.
    //       Each kernel need to store the pallet it was compiled with.
    char palette[64] = {};
};

class BrgemmKernelExecutor : public CPUKernelExecutor<BrgemmKernelConfig, BrgemmCompiledKernel> {
public:
    struct call_args {
        const void* A = nullptr;
        const void* B = nullptr;
        void* C = nullptr;
        void* scratch = nullptr;
        amx_tile_config_t* amx_tile_config = nullptr;
    };
    BrgemmKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache,
                         const std::shared_ptr<BrgemmKernelConfig>& config);

    /** Function that will be called in runtime to execute the kernel */
    static void execute(const BrgemmKernelExecutor* executor, call_args* args);

protected:
    std::shared_ptr<BrgemmCompiledKernel> compile_kernel(const std::shared_ptr<const BrgemmKernelConfig>& c) const override;
    void update_config(const ov::snippets::lowered::ExpressionPtr& expr, std::shared_ptr<BrgemmKernelConfig>& config) const override;
};
#define GET_OFF_BRGEMM_ARGS(field) offsetof(BrgemmKernelExecutor::call_args, field)
}   // namespace intel_cpu
}   // namespace ov
