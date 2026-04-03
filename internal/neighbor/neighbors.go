package neighbor

import (
	"container/heap"
)

func GetKNearest(k int, target []float64, trainX [][]float64, measure func(a, b []float64) (float64, error)) ([]int, error) {
	h := &MaxHeapNeighbor{}
	heap.Init(h)

	for i, row := range trainX {
		d, err := measure(target, row)
		if err != nil {
			return nil, err
		}

		item := NeighborItem{
			Index:    i,
			Distance: d,
		}

		if h.Len() < k {
			heap.Push(h, item)
		} else if d < (*h)[0].Distance {
			heap.Pop(h)
			heap.Push(h, item)
		}
	}

	indices := make([]int, h.Len())
	for i := h.Len() - 1; i >= 0; i-- {
		item := heap.Pop(h).(NeighborItem)
		indices[i] = item.Index
	}

	return indices, nil
}
