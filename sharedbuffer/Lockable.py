from dataclasses import dataclass, field

@dataclass
class Lockable:
    element: any
    ref_counter: int = field(default=0)

    def lock(self):
        self.ref_counter += 1

    def release(self) -> bool:
        if self.ref_counter <= 0:
            return True
        self.ref_counter -= 1
        return self.ref_counter == 0

    def get_element(self):
        return self.element

    def get_ref_counter(self) -> int:
        return self.ref_counter

    def __str__(self) -> str:
        return f"Lock{{ref_counter={self.ref_counter}}}"

    def __eq__(self, other) -> bool:
        if isinstance(other, Lockable):
            return self.ref_counter == other.ref_counter and self.element == other.element
        return False

    def __hash__(self) -> int:
        return hash((self.ref_counter, self.element))


if __name__ == "__main__":
    # 示例使用
    lockable = Lockable(element="test_element", ref_counter=0)
    print(lockable)  # 输出: Lock{ref_counter=0}

    lockable.lock()
    print(lockable.get_ref_counter())  # 输出: 1

    lockable.release()
    print(lockable.get_ref_counter())  # 输出: 0

    print(lockable.get_element())  # 输出: test_element

    lockable2 = Lockable(element="test_element", ref_counter=0)
    print(lockable == lockable2)  # 输出: True

    lockable3 = Lockable(element="another_element", ref_counter=1)
    print(lockable == lockable3)  # 输出: False

    print(hash(lockable))  # 输出: 哈希值
    print(hash(lockable2))  # 输出: 相同的哈希值
    print(hash(lockable3))  # 输出: 不同的哈希值