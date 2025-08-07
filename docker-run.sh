#!/bin/bash

# YouTube Boat Scraper Docker Management Script with CUDA/GPU Support

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_gpu() {
    echo -e "${PURPLE}[GPU]${NC} $1"
}

# Check if Docker is running
check_docker() {
    print_status "Checking Docker..."
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
    print_success "Docker is running"
}

# Check GPU and NVIDIA Docker support
check_gpu_support() {
    print_status "Checking GPU support..."
    
    # Check if nvidia-smi is available on host
    if command -v nvidia-smi >/dev/null 2>&1; then
        print_gpu "NVIDIA GPU detected on host:"
        nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader,nounits | while IFS=, read -r name memory; do
            print_gpu "  - $name (${memory}MB VRAM)"
        done
        
        # Check Docker GPU runtime support
        if docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
            print_success "Docker GPU support is working"
            GPU_AVAILABLE=true
        else
            print_warning "Docker GPU support not available"
            print_warning "Please install nvidia-docker2 or Docker with GPU support"
            print_warning "See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            GPU_AVAILABLE=false
        fi
    else
        print_warning "No NVIDIA GPU detected on host"
        print_warning "Container will run in CPU-only mode"
        GPU_AVAILABLE=false
    fi
}

# Check if required files exist
check_files() {
    print_status "Checking required files..."
    
    if [ ! -f "Dockerfile" ]; then
        print_error "Dockerfile not found in current directory"
        exit 1
    fi
    
    if [ ! -f "docker-compose.yml" ]; then
        print_error "docker-compose.yml not found in current directory"
        exit 1
    fi
    
    if [ ! -d "src" ]; then
        print_error "src directory not found"
        exit 1
    fi
    
    if [ ! -d "resources" ]; then
        print_error "resources directory not found"
        exit 1
    fi
    
    if [ ! -f "resources/config.yaml" ]; then
        print_warning "config.yaml not found in resources directory"
    fi
    
    if [ ! -f "resources/credentials.json" ]; then
        print_warning "credentials.json not found in resources directory"
        print_warning "GCS functionality may not work"
    fi
    
    print_success "File check completed"
}

# Create data directory if it doesn't exist
setup_directories() {
    print_status "Setting up directories..."
    mkdir -p ./data
    print_success "Directories created"
}

# Build the Docker image
build_image() {
    print_status "Building Docker image with CUDA support..."
    print_status "This may take several minutes on first build..."
    
    # Build with progress output
    docker-compose build --no-cache --progress=plain
    
    print_success "Docker image built successfully"
    
    # Test GPU in built image
    if [ "$GPU_AVAILABLE" = true ]; then
        print_status "Testing GPU access in container..."
        if docker run --rm --gpus all youtube-boat-scraper-gpu:latest python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null; then
            print_success "GPU access confirmed in container"
        else
            print_warning "GPU not accessible in container (will run on CPU)"
        fi
    fi
}

# Run the container
run_container() {
    print_status "Starting YouTube Boat Scraper container with GPU support..."
    
    if [ "$GPU_AVAILABLE" = true ]; then
        print_gpu "Container will use GPU acceleration"
        docker-compose up -d
    else
        print_warning "Container will run in CPU-only mode"
        # You could modify docker-compose.yml or use different service here
        docker-compose up -d
    fi
    
    print_success "Container started successfully"
    
    # Show initial logs to verify startup
    sleep 3
    print_status "Initial startup logs:"
    docker-compose logs --tail=20
    
    echo ""
    print_status "Commands:"
    print_status "  View logs: $0 logs"
    print_status "  Stop: $0 stop"
    print_status "  Monitor GPU: watch nvidia-smi"
}

# View logs
view_logs() {
    print_status "Showing container logs (Ctrl+C to exit)..."
    docker-compose logs -f
}

# Stop container
stop_container() {
    print_status "Stopping container..."
    docker-compose down
    print_success "Container stopped"
}

# Monitor GPU usage
monitor_gpu() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        print_gpu "Starting GPU monitoring (Ctrl+C to exit)..."
        print_gpu "This will show GPU usage while your container runs"
        watch -n 1 'nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv'
    else
        print_error "nvidia-smi not available. Cannot monitor GPU usage."
    fi
}

# System info
show_system_info() {
    echo "======================================"
    echo "System Information"
    echo "======================================"
    
    print_status "Docker version:"
    docker --version
    
    print_status "Docker Compose version:"
    docker-compose --version
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        print_gpu "NVIDIA Driver version:"
        nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1
        
        print_gpu "GPU Information:"
        nvidia-smi --query-gpu=gpu_name,memory.total,compute_cap --format=csv,noheader
    else
        print_warning "No NVIDIA GPU detected"
    fi
    
    print_status "Available disk space:"
    df -h . | tail -1
    
    print_status "Available memory:"
    free -h
}

# Performance test
run_performance_test() {
    print_status "Running CUDA performance test..."
    
    if [ "$GPU_AVAILABLE" = true ]; then
        print_gpu "Testing GPU performance..."
        docker run --rm --gpus all nvidia/cuda:12.1-devel-ubuntu22.04 bash -c "
            apt-get update >/dev/null 2>&1
            apt-get install -y python3 python3-pip >/dev/null 2>&1
            pip3 install torch --index-url https://download.pytorch.org/whl/cu121 >/dev/null 2>&1
            python3 -c \"
import torch
import time

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
    
    # Simple performance test
    size = 5000
    a = torch.randn(size, size).to(device)
    b = torch.randn(size, size).to(device)
    
    # Warmup
    for _ in range(3):
        c = torch.matmul(a, b)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()
    
    print(f'GPU Matrix multiplication time: {(end-start)/10:.4f} seconds per operation')
    print(f'GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB')
    print(f'GPU Memory cached: {torch.cuda.memory_reserved()/1024**2:.1f} MB')
else:
    print('CUDA not available')
\"
        "
    else
        print_warning "GPU not available for performance test"
    fi
}

# Main function
main() {
    echo "================================================="
    echo "YouTube Boat Scraper Docker Manager (GPU Edition)"
    echo "================================================="
    
    check_docker
    check_gpu_support
    check_files
    setup_directories
    
    case "${1:-run}" in
        "build")
            build_image
            ;;
        "run")
            build_image
            run_container
            ;;
        "logs")
            view_logs
            ;;
        "stop")
            stop_container
            ;;
        "restart")
            stop_container
            sleep 2
            build_image
            run_container
            ;;
        "clean")
            print_status "Cleaning up Docker resources..."
            docker-compose down
            docker system prune -f
            print_success "Cleanup completed"
            ;;
        "gpu")
            monitor_gpu
            ;;
        "info")
            show_system_info
            ;;
        "test")
            run_performance_test
            ;;
        *)
            echo "Usage: $0 [build|run|logs|stop|restart|clean|gpu|info|test]"
            echo ""
            echo "Commands:"
            echo "  build   - Build Docker image with CUDA support"
            echo "  run     - Build and run container with GPU (default)"
            echo "  logs    - View container logs"
            echo "  stop    - Stop running container"
            echo "  restart - Stop, rebuild, and start container"
            echo "  clean   - Stop container and clean Docker resources"
            echo "  gpu     - Monitor GPU usage in real-time"
            echo "  info    - Show system and GPU information"
            echo "  test    - Run CUDA performance test"
            echo ""
            echo "GPU Support:"
            echo "  - Requires NVIDIA GPU with CUDA support"
            echo "  - Requires nvidia-docker2 or Docker with GPU support"
            echo "  - Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            exit 1
            ;;
    esac
}

main "$@"