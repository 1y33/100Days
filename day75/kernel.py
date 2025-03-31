import torch
import triton
import triton.language as tl

@triton.jit
def knowledge_distillation_loss_kernel(
    student_logits_ptr, 
    teacher_logits_ptr,
    labels_ptr,
    output_ptr,
    batch_size,
    num_classes,
    temperature: tl.float32,
    alpha: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size
    ce_loss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    kd_loss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, BLOCK_SIZE):
        if not mask[i]:
            continue
        offset = offsets[i]
        base_offset = offset * num_classes
        label = tl.load(labels_ptr + offset)
        student_max = tl.float32(-float('inf'))
        for c in range(0, num_classes):
            student_val = tl.load(student_logits_ptr + base_offset + c) 
            student_max = tl.maximum(student_max, student_val)
        student_sum = 0.0
        student_probs = tl.zeros([num_classes], dtype=tl.float32)
        for c in range(0, num_classes):
            student_val = tl.load(student_logits_ptr + base_offset + c)
            prob = tl.exp(student_val - student_max)
            student_probs[c] = prob
            student_sum += prob
        for c in range(0, num_classes):
            student_probs[c] = student_probs[c] / student_sum
        ce_loss[i] = -tl.log(student_probs[label] + 1e-12)
        student_t_sum = 0.0
        teacher_t_sum = 0.0
        student_t_probs = tl.zeros([num_classes], dtype=tl.float32)
        teacher_t_probs = tl.zeros([num_classes], dtype=tl.float32)
        student_t_max = tl.float32(-float('inf'))
        teacher_t_max = tl.float32(-float('inf'))
        for c in range(0, num_classes):
            student_val = tl.load(student_logits_ptr + base_offset + c) / temperature
            teacher_val = tl.load(teacher_logits_ptr + base_offset + c) / temperature
            student_t_max = tl.maximum(student_t_max, student_val)
            teacher_t_max = tl.maximum(teacher_t_max, teacher_val)
        for c in range(0, num_classes):
            student_val = tl.load(student_logits_ptr + base_offset + c) / temperature
            teacher_val = tl.load(teacher_logits_ptr + base_offset + c) / temperature
            student_exp = tl.exp(student_val - student_t_max)
            teacher_exp = tl.exp(teacher_val - teacher_t_max)
            student_t_probs[c] = student_exp
            teacher_t_probs[c] = teacher_exp
            student_t_sum += student_exp
            teacher_t_sum += teacher_exp
        kld = 0.0
        for c in range(0, num_classes):
            student_prob = student_t_probs[c] / student_t_sum
            teacher_prob = teacher_t_probs[c] / teacher_t_sum
            if teacher_prob > 1e-12:
                kld += teacher_prob * tl.log(teacher_prob / (student_prob + 1e-12))
        kd_loss[i] = kld * (temperature * temperature)
    for i in range(0, BLOCK_SIZE):
        if mask[i]:
            total_loss = alpha * ce_loss[i] + (1.0 - alpha) * kd_loss[i]
            tl.store(output_ptr + offsets[i], total_loss)

def knowledge_distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    batch_size, num_classes = student_logits.shape
    output = torch.zeros(batch_size, device=student_logits.device, dtype=torch.float32)
    BLOCK_SIZE = 32
    grid = (triton.cdiv(batch_size, BLOCK_SIZE),)
    knowledge_distillation_loss_kernel[grid](
        student_logits, 
        teacher_logits,
        labels,
        output,
        batch_size,
        num_classes,
        temperature,
        alpha,
        BLOCK_SIZE,
    )
    return output.mean()

if __name__ == "__main__":
    batch_size = 64
    num_classes = 10
    student_logits = torch.randn(batch_size, num_classes, device='cuda')
    teacher_logits = torch.randn(batch_size, num_classes, device='cuda')
    labels = torch.randint(0, num_classes, (batch_size,), device='cuda')
    loss = knowledge_distillation_loss(
        student_logits, 
        teacher_logits, 
        labels,
        temperature=2.0, 
        alpha=0.5
    )
    print(f"Knowledge Distillation Loss: {loss.item()}")
