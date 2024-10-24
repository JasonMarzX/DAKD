import numpy as np
from scipy.special import softmax
from scipy.stats import entropy

def calculate_distillation_loss(teacher_probs, student_probs):
    kl_divergence = entropy(teacher_probs.T, student_probs.T).mean()  
    return kl_divergence

def train_student_model(student_model, teacher_model, dataset, temperature=3.0, alpha=0.7):
    teacher_logits = np.random.rand(len(dataset), 10)  
    teacher_probs = softmax(teacher_logits / temperature, axis=1)  

    student_logits = np.random.rand(len(dataset), 10) 
    student_probs = softmax(student_logits / temperature, axis=1)  

    distillation_loss = calculate_distillation_loss(teacher_probs, student_probs)
    
    true_labels = np.eye(10)[np.random.choice(10, len(dataset))] 
    hard_loss = -np.sum(true_labels * np.log(student_probs), axis=1).mean() 
    
    total_loss = alpha * distillation_loss + (1 - alpha) * hard_loss
    
    print(f"Distillation loss: {distillation_loss}, Hard loss: {hard_loss}, Total loss: {total_loss}")
    
    student_model += teacher_model * (1 - total_loss)
    
    return student_model

def adaptive_teacher_selection(dataset, teacher_model, student_model, epochs, save_interval):
    saved_teacher_models = [] 
    loss_list = [] 
    
    for k in range(1, epochs // save_interval + 1):
        trained_teacher_model = train_teacher_model(teacher_model, dataset, k * save_interval)
        saved_teacher_models.append(trained_teacher_model)
        print(f"Saved teacher model at stage {k}, model state: {trained_teacher_model}")
    
    for k, saved_teacher in enumerate(saved_teacher_models):
        data_subset = dataset[:len(dataset) // 10] 
        trained_student_model = train_student_model(student_model, saved_teacher, data_subset)
        
        student_probs = softmax(np.random.rand(len(data_subset), 10), axis=1)  
        teacher_probs = softmax(np.random.rand(len(data_subset), 10), axis=1) 
        distillation_loss = calculate_distillation_loss(teacher_probs, student_probs)
        loss_list.append(distillation_loss)
        print(f"Distillation loss for teacher model at stage {k}: {distillation_loss}")
    
    best_teacher_index = np.argmin(loss_list)
    best_teacher_model = saved_teacher_models[best_teacher_index]
    
    print(f"Best teacher model selected at stage {best_teacher_index} with loss {loss_list[best_teacher_index]}")
    return best_teacher_model, best_teacher_index
