template<typename T>
class LockFreeList {
    struct Node {
        T data;
        std::atomic<Node *> next{nullptr};
    };
    Node *base = new Node();
    std::atomic<Node *> head{base};
    std::atomic<Node *> tail{base};
    std::atomic<int> size{0};

public:
    bool add(const T &data) {
        Node *node = new Node;
        node->data = data;
        node->next.store(nullptr);
        Node *p = nullptr;
        Node *pNull = nullptr;
        do {
            p = tail.load();
        } while (p->next.compare_exchange_weak(pNull, node) == false);
        tail.store(node);
        size++;
        return true;
        /*Node *null = nullptr;
        Node *newNode = new Node;
        newNode->data = data;
        newNode->next.store(nullptr);
        for (Node *t = tail.load(), p = t;;) {
            // 获取尾节点的下一个节点
            Node *q = p->next.load();
            if (q == nullptr) {// p就是尾节点
                // next为空就设置为新节点
                if (p->next.compare_exchange_weak(null, newNode)) {// 成功
                    // 首次添加不指定新的尾节点
                    if (p != t)
                        // 尾节点为t，就把尾节点设置为新节点
                        tail.compare_exchange_weak(t, newNode);
                    return true;
                }
            }
                // 头节点移除后调用updateHead方法
                // updateHead会把原来头节点下级指向自己
            else if (p == q)
                // 从头走
                p = (t != (t = tail.load())) ? t : head.load();
            else
                // 推动p节点后移
                p = (p != t && t != (t = tail.load())) ? t : q;
        }*/
    }

    /*void updateHead(Node *&h, Node *&p) {
        // 头节点修改为p
        if (h != p && head.compare_exchange_weak(h, p))
            // 原头节点的下级节点修改为自己
            h->next.store(h);
    }*/

    bool remove(T &data) {
        Node *oldp = nullptr;
        Node *p = nullptr;
        do {
            oldp = head.load();
            p = oldp->next.load();
            if (p == nullptr) {
                return false;
            }
        } while (head.compare_exchange_weak(oldp, p) == false);
        data = p->data;
        delete oldp;
        size--;
        return true;
        /*restartFromHead:
        for (;;) {
            for (Node *h = head.load(), p = h, q;;) {
                // 头节点不为null，先把头节点置为null
                if (h->next.load() != nullptr && head.compare_exchange_weak(h, nullptr)) {
                    data = h->data;
                    // 第一次操作成功p=h，后面线程修改头节点
                    if (p != h)
                        // 修改头节点
                        updateHead(h, ((q = p->next.load()) != nullptr) ? q : p);
                    delete h;
                    return true;
                }
                    // 没获取到锁或者为空
                else if ((q = p.next) == nullptr) {
                    // 下级节点为空修改头节点
                    updateHead(h, p);
                    return false;
                }
                    // 头节点移除后调用updateHead方法
                    // updateHead会把原来头节点下级指向自己
                else if (p == q)
                    goto restartFromHead;
                else
                    // p指向下级节点
                    p = q;
            }
        }*/
    }

    bool empty() {
        return head.load()->next.load() == nullptr;
    }
};
