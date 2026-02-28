package mathx

type MaxHeapNeighbor []NeighborItem

func (h MaxHeapNeighbor) Len() int {
	return len(h)
}

func (h MaxHeapNeighbor) Less(i, j int) bool {
	return h[i].Distance > h[j].Distance
}

func (h MaxHeapNeighbor) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *MaxHeapNeighbor) Push(x interface{}) {
	*h = append(*h, x.(NeighborItem))
}

func (h *MaxHeapNeighbor) Pop() interface{} {
	old := *h
	n := len(old)

	item := old[n-1]

	*h = old[0 : n-1]

	return item
}
