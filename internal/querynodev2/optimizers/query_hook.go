package optimizers

// QueryHook is the interface for search/query parameter optimizer.
type QueryHook interface {
	Optimize2(map[string]any) ([]int64, []int64, error)
	Optimize(map[string]any) ([]string, []int64, error)
	Run(map[string]any) error
	Init(string) error
	InitTuningConfig(map[string]string) error
	DeleteTuningConfig(string) error
}
