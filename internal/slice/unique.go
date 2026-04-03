package slice

func UniqueInts(integers []int) []int {
	m := make(map[int]struct{})

	for _, integer := range integers {
		m[integer] = struct{}{}
	}

	unique := make([]int, 0, len(m))
	for key := range m {
		unique = append(unique, key)
	}

	return unique
}
