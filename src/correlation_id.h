// Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <string>

#include "pb_string.h"
#include "pb_utils.h"

namespace triton { namespace backend { namespace python {

enum class CorrIdDataType { UINT64, STRING };

struct SequenceIdShm {
  bi::managed_external_buffer::handle_t sequence_label_shm_handle;
  uint64_t sequence_index;
  CorrIdDataType id_type;
};

class SequenceId {
 public:
  SequenceId();
  SequenceId(const std::string& sequence_label);
  SequenceId(uint64_t sequence_index);
  SequenceId(const SequenceId& rhs);
  // SequenceId(const std::unique_ptr<SequenceId>& rhs_ptr);
  SequenceId& operator=(const SequenceId& rhs);

  // ~SequenceId(){};

  /// Save SequenceId object to shared memory.
  /// \param shm_pool Shared memory pool to save the SequenceId object.
  void SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool);

  /// Create a SequenceId object from shared memory.
  /// \param shm_pool Shared memory pool
  /// \param handle Shared memory handle of the SequenceId.
  /// \return Returns the SequenceId in the specified request_handle
  /// location.
  static std::unique_ptr<SequenceId> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t handle);

  // Functions that help determine exact type of sequence Id
  CorrIdDataType Type() const { return id_type_; }
  bool InSequence() const
  {
    return ((sequence_label_ != "") || (sequence_index_ != 0));
  }

  // Get the value of the SequenceId based on the type
  const std::string& StringValue() const { return sequence_label_; }
  uint64_t UnsignedIntValue() const { return sequence_index_; }

  bi::managed_external_buffer::handle_t ShmHandle() { return shm_handle_; }

 private:
  // The private constructor for creating a SequenceId object from shared
  // memory.
  SequenceId(
      AllocatedSharedMemory<SequenceIdShm>& sequence_id_shm,
      std::unique_ptr<PbString>& sequence_label_shm);

  std::string sequence_label_;
  uint64_t sequence_index_;
  CorrIdDataType id_type_;

  // Shared Memory Data Structures
  AllocatedSharedMemory<SequenceIdShm> sequence_id_shm_;
  SequenceIdShm* sequence_id_shm_ptr_;
  bi::managed_external_buffer::handle_t shm_handle_;
  std::unique_ptr<PbString> sequence_label_shm_;
};

}}};  // namespace triton::backend::python
