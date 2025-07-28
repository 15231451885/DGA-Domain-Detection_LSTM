"""
对抗样本生成模块：通过轻微扰动正常域名生成对抗样本
功能：
1. 字符随机替换（如 a -> 4）
2. 随机插入/删除字符
3. 控制扰动强度（最大修改字符数）
"""
import random

class AdversarialGenerator:
    def __init__(self, perturbation_strength=0.3):
        """
        :param perturbation_strength: 扰动强度（0-1），控制修改字符的比例
        """
        self.perturbation_strength = perturbation_strength
        self.char_substitutions = {
            'a': ['4', '@'],
            'e': ['3'],
            'i': ['1', '!'],
            'o': ['0'],
            's': ['5', '$']
        }

    def generate(self, domain_name):
        """生成对抗样本"""
        chars = list(domain_name)
        max_changes = max(1, int(len(chars) * self.perturbation_strength))
        
        for _ in range(random.randint(1, max_changes)):
            idx = random.randint(0, len(chars)-1)
            original_char = chars[idx]
            
            # 50%概率替换，25%概率插入，25%概率删除
            action = random.random()
            
            if action < 0.5 and original_char in self.char_substitutions:
                # 字符替换
                chars[idx] = random.choice(self.char_substitutions[original_char])
            elif action < 0.75:
                # 插入随机字符
                chars.insert(idx, random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789')))
            else:
                # 删除字符（至少保留3个字符）
                if len(chars) > 3:
                    chars.pop(idx)
        
        return ''.join(chars)

    def batch_generate(self, domain_names, labels=None):
        """批量生成对抗样本"""
        adversarial_examples = []
        for name in domain_names:
            adversarial_examples.append(self.generate(name))
        return adversarial_examples