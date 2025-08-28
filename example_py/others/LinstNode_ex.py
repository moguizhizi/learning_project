class ListNode():
    def __init__(self, value, next=None):
        self.value = value
        self.next = next
        
def delete_all_duplicates(head:ListNode):
    

    dumpy = ListNode(-100000)
    save_point = dumpy
    current_point:ListNode = head
    
    
    while current_point:
        next_point:ListNode = current_point.next
        is_save = True
        while next_point and next_point.value == current_point.value:
            next_point = next_point.next
            is_save = False
        
        if is_save == False:
            current_point = next_point
        else:
            save_point.next = current_point
            save_point = current_point
            current_point = current_point.next
            save_point.next = None
    
    return dumpy.next
        
    
    

def main():
    
    head = ListNode(0)
    node = head
    temp = ListNode(0)
    node.next = temp
    node = temp
    
    temp = ListNode(1)
    node.next = temp
    node = temp
    
    temp = ListNode(1)
    node.next = temp
    node = temp
    
    temp = ListNode(2)
    node.next = temp
    node = temp
    
    temp = ListNode(3)
    node.next = temp
    node = temp
    
    temp = ListNode(3)
    node.next = temp
    node = temp
    
    temp = ListNode(4)
    node.next = temp
    node = temp
    
    temp = ListNode(5)
    node.next = temp
    node = temp
    
    temp = ListNode(5)
    node.next = temp
    node = temp
    
    current = head
    while current:
        print(current.value)
        current = current.next
        
        
    print("*"*10)
        
    save_point = delete_all_duplicates(head)
    current = save_point
    while current:
        print(current.value)
        current = current.next
        
    

main()