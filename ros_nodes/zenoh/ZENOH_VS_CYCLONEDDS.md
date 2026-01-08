# Zenoh vs CycloneDDS for Video Streaming

## Executive Summary

**For compressed video streaming (<1MB frames), CycloneDDS is recommended.**

Based on benchmark results, the performance difference is minimal (<0.1ms latency difference), but CycloneDDS has:
- Slightly lower latency for compressed images
- No additional setup required (default ROS2 middleware)
- Simpler deployment
- Better compatibility with standard ROS2 tools

**Zenoh is worth considering if you need:**
- Raw/uncompressed streaming (Zenoh: 1.4ms vs CycloneDDS: 8.2ms)
- Multi-hop networks or complex topologies
- Large-scale deployments (100+ nodes)
- Advanced discovery mechanisms

## Benchmark Results Analysis

### Compressed Video Streaming (<1MB frames)

**JPEG Q85, Quarter Resolution (160x120) - ~1.8 Mbps:**
- **Zenoh**: 0.444ms avg latency, 0.561ms P95
- **CycloneDDS**: 0.423ms avg latency, 0.523ms P95
- **Difference**: CycloneDDS is ~5% faster

**JPEG Q75, Full Resolution (640x480) - ~10 Mbps:**
- **Zenoh**: 0.384ms avg latency, 0.439ms P95
- **CycloneDDS**: 0.345ms avg latency, 0.390ms P95
- **Difference**: CycloneDDS is ~10% faster

**Key Finding**: For compressed video (<1MB frames), both middlewares perform excellently with sub-millisecond latency. The difference is negligible for practical purposes.

### Raw/Uncompressed Streaming

**Raw Full Resolution (640x480) - ~221 Mbps:**
- **Zenoh**: 1.441ms avg latency, 1.639ms P95
- **CycloneDDS**: 8.203ms avg latency, 12.364ms P95
- **Difference**: Zenoh is **5.7x faster** for raw images

**Conclusion**: Zenoh has a significant advantage for uncompressed streaming, but this is rarely needed for video applications.

## Recommendations by Use Case

### ✅ Use CycloneDDS (Default) If:

1. **Compressed video streaming** (your use case)
   - JPEG compression keeps frames <1MB
   - Minimal latency difference
   - No additional setup

2. **Standard ROS2 deployment**
   - Works out of the box
   - Better tool compatibility
   - Easier troubleshooting

3. **Single network, simple topology**
   - Direct publisher-subscriber
   - Local network only

4. **Development and testing**
   - Faster iteration
   - Standard ROS2 tools work better

### ✅ Use Zenoh If:

1. **Raw/uncompressed streaming**
   - Significant latency advantage (5.7x faster)
   - High bandwidth available
   - Quality requirements demand raw

2. **Complex network topologies**
   - Multi-hop networks
   - WAN deployments
   - Advanced routing needs

3. **Large-scale deployments**
   - 100+ nodes
   - Better discovery mechanisms
   - More efficient resource usage

4. **Special requirements**
   - Custom transport protocols
   - Advanced QoS features
   - Integration with Zenoh ecosystem

## Performance Comparison Summary

| Metric | Compressed (<1MB) | Raw (>1MB) |
|--------|------------------|------------|
| **Latency (avg)** | CycloneDDS ~5-10% faster | Zenoh 5.7x faster |
| **P95 Latency** | CycloneDDS ~5-10% faster | Zenoh 7.5x faster |
| **Bandwidth** | Identical | Identical |
| **Setup Complexity** | CycloneDDS simpler | Zenoh requires setup |
| **Compatibility** | CycloneDDS better | Similar |

## Practical Recommendation

**For your video streaming use case:**

1. **Stick with CycloneDDS (default)**
   - You're using compressed video (<1MB frames)
   - Performance difference is negligible (<0.1ms)
   - Simpler setup and maintenance
   - Better ROS2 tool compatibility

2. **Use JPEG compression**
   - Q75-Q85 quality provides good balance
   - Keeps frames <1MB (minimal latency impact)
   - Reduces bandwidth significantly

3. **Consider Zenoh only if:**
   - You need raw streaming (unlikely for video)
   - You have complex network requirements
   - You're scaling to 100+ nodes

## Migration Path

If you're currently using Zenoh and want to switch to CycloneDDS:

```bash
# Simply use the default middleware (no setup needed)
source /opt/ros/jazzy/setup.bash
# No need to source setup_zenoh.sh
```

If you want to test both:

```bash
# Test CycloneDDS
source setup_default.sh
python3 camera_publisher.py --source 0 --jpeg-quality 85

# Test Zenoh
source setup_zenoh.sh
python3 camera_publisher.py --source 0 --jpeg-quality 85
```

## Conclusion

For compressed video streaming, **CycloneDDS is the better choice**:
- ✅ Slightly better performance
- ✅ Simpler setup
- ✅ Better compatibility
- ✅ No additional dependencies

The performance difference is so small (<0.1ms) that it's not worth the added complexity of Zenoh for this use case. Save Zenoh for when you actually need its advantages (raw streaming, complex topologies, large scale).

