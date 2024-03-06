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

#include "correlation_id.h"

namespace triton { namespace backend { namespace python {

SequenceId::SequenceId()
    : sequence_label_(""), sequence_index_(0),
      id_type_(CorrelationIdDataType::UINT64)
{
}

SequenceId::SequenceId(const std::string& sequence_label)
    : sequence_label_(sequence_label), sequence_index_(0),
      id_type_(CorrelationIdDataType::STRING)
{
}

SequenceId::SequenceId(uint64_t sequence_index)
    : sequence_label_(""), sequence_index_(sequence_index),
      id_type_(CorrelationIdDataType::UINT64)
{
}

SequenceId::SequenceId(const SequenceId& rhs)
{
  sequence_index_ = rhs.sequence_index_;
  id_type_ = rhs.id_type_;
  sequence_label_ = rhs.sequence_label_;
}

SequenceId&
SequenceId::operator=(const SequenceId& rhs)
{
  sequence_index_ = rhs.sequence_index_;
  id_type_ = rhs.id_type_;
  sequence_label_ = rhs.sequence_label_;
  return *this;
}

void
SequenceId::SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool)
{
  AllocatedSharedMemory<SequenceIdShm> sequence_id_shm =
      shm_pool->Construct<SequenceIdShm>();
  sequence_id_shm_ptr_ = sequence_id_shm.data_.get();

  std::unique_ptr<PbString> sequence_label_shm =
      PbString::Create(shm_pool, sequence_label_);

  sequence_id_shm_ptr_->sequence_index = sequence_index_;
  sequence_id_shm_ptr_->sequence_label_shm_handle =
      sequence_label_shm->ShmHandle();
  sequence_id_shm_ptr_->id_type = id_type_;

  // Save the references to shared memory.
  sequence_id_shm_ = std::move(sequence_id_shm);
  sequence_label_shm_ = std::move(sequence_label_shm);
  shm_handle_ = sequence_id_shm_.handle_;
}

std::unique_ptr<SequenceId>
SequenceId::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t handle)
{
  AllocatedSharedMemory<SequenceIdShm> sequence_id_shm =
      shm_pool->Load<SequenceIdShm>(handle);
  SequenceIdShm* sequence_id_shm_ptr = sequence_id_shm.data_.get();

  std::unique_ptr<PbString> sequence_label_shm = PbString::LoadFromSharedMemory(
      shm_pool, sequence_id_shm_ptr->sequence_label_shm_handle);

  return std::unique_ptr<SequenceId>(
      new SequenceId(sequence_id_shm, sequence_label_shm));
}

SequenceId::SequenceId(
    AllocatedSharedMemory<SequenceIdShm>& sequence_id_shm,
    std::unique_ptr<PbString>& sequence_label_shm)
    : sequence_id_shm_(std::move(sequence_id_shm)),
      sequence_label_shm_(std::move(sequence_label_shm))
{
  sequence_id_shm_ptr_ = sequence_id_shm_.data_.get();
  sequence_label_ = sequence_label_shm_->String();
  sequence_index_ = sequence_id_shm_ptr_->sequence_index;
  id_type_ = sequence_id_shm_ptr_->id_type;
}

}}};  // namespace triton::backend::python
