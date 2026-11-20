// vk_dot.c (2026-11-19, LLM engine G1): int8 dot product on the Mali-G52 through panvk.
// Refuses any device whose name lacks "Mali". Run only under no-npu-guard.so (panvk probes the NPU node).
//   vk_dot check dot_check.spv            c[i] = sdot(a[i], b[i]) + 7 over 65536 random words, vs the CPU
//   vk_dot bench SHADER.spv ROUNDS REPS    dot_bench / emul_bench: 8 dot chains per invocation, vs the CPU, timed
// Build: gcc -O2 -o vk_dot vk_dot.c -lvulkan
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "Vulkan error %d at %s\n", r_, #x); exit(2); } } while (0)

static VkDevice dev;
static VkPhysicalDevice pd;
static VkQueue queue;
static uint32_t qf = UINT32_MAX;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static int8_t sb(int32_t v, int k) { return (int8_t)((uint32_t)v >> (8 * k)); }
static int32_t sdot(int32_t x, int32_t y) {
   int32_t s = 0;
   for (int k = 0; k < 4; k++) s += (int32_t)sb(x, k) * sb(y, k);
   return s;
}

typedef struct { VkBuffer b; VkDeviceMemory m; void *p; } Buf;

static Buf mkbuf(size_t bytes) {
   Buf r;
   VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes,
                            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
   CHECK(vkCreateBuffer(dev, &bi, NULL, &r.b));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, r.b, &mr);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   uint32_t want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, t = 0;
   for (; t < mp.memoryTypeCount; t++)
      if ((mr.memoryTypeBits & (1u << t)) && (mp.memoryTypes[t].propertyFlags & want) == want) break;
   VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = t};
   CHECK(vkAllocateMemory(dev, &ai, NULL, &r.m));
   CHECK(vkBindBufferMemory(dev, r.b, r.m, 0));
   CHECK(vkMapMemory(dev, r.m, 0, bytes, 0, &r.p));
   return r;
}

int main(int argc, char **argv) {
   if (argc < 3) { fprintf(stderr, "usage: vk_dot check|bench SHADER.spv [ROUNDS REPS]\n"); return 1; }
   int bench = !strcmp(argv[1], "bench");
   int rounds = argc > 3 ? atoi(argv[3]) : 256, reps = argc > 4 ? atoi(argv[4]) : 5;
   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1};
   VkInstanceCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
   VkInstance inst;
   CHECK(vkCreateInstance(&ii, NULL, &inst));
   uint32_t n = 8;
   VkPhysicalDevice pds[8];
   CHECK(vkEnumeratePhysicalDevices(inst, &n, pds));
   VkPhysicalDeviceProperties props;
   pd = VK_NULL_HANDLE;
   for (uint32_t i = 0; i < n; i++) {
      vkGetPhysicalDeviceProperties(pds[i], &props);
      if (strstr(props.deviceName, "Mali")) { pd = pds[i]; break; }
   }
   if (pd == VK_NULL_HANDLE) { fprintf(stderr, "no Mali device among %u: refusing\n", n); return 2; }
   VkPhysicalDeviceDriverProperties drv = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
   VkPhysicalDeviceShaderIntegerDotProductProperties dp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES, .pNext = &drv};
   VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &dp};
   vkGetPhysicalDeviceProperties2(pd, &p2);
   printf("device: %s, driver: %s %s, 4x8 packed signed dot accelerated: %s\n", props.deviceName, drv.driverName,
          drv.driverInfo, dp.integerDotProduct4x8BitPackedSignedAccelerated ? "yes" : "no");

   uint32_t nq = 8;
   VkQueueFamilyProperties qp[8];
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qp);
   for (uint32_t i = 0; i < nq; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
   float prio = 1;
   VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio};
   VkPhysicalDeviceShaderIntegerDotProductFeatures df = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES, .shaderIntegerDotProduct = VK_TRUE};
   const char *ext[] = {"VK_KHR_shader_integer_dot_product"};
   VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &df, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi,
                            .enabledExtensionCount = 1, .ppEnabledExtensionNames = ext};
   CHECK(vkCreateDevice(pd, &di, NULL, &dev));
   vkGetDeviceQueue(dev, qf, 0, &queue);

   const uint32_t inv = 65536, words = bench ? 8 * inv : inv;
   Buf A = mkbuf(words * 4), B = mkbuf(words * 4), C = mkbuf(inv * 4);
   int32_t *a = A.p, *b = B.p, *c = C.p;
   srand(61119);
   for (uint32_t i = 0; i < words; i++) { a[i] = (int32_t)((uint32_t)rand() * 2654435761u); b[i] = (int32_t)((uint32_t)rand() * 40503u + rand()); }
   memset(c, 0, inv * 4);

   FILE *f = fopen(argv[2], "rb");
   if (!f) { perror(argv[2]); return 1; }
   static uint32_t code[1 << 16];
   size_t cs = fread(code, 1, sizeof code, f);
   fclose(f);
   VkShaderModuleCreateInfo smi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = cs, .pCode = code};
   VkShaderModule sm;
   CHECK(vkCreateShaderModule(dev, &smi, NULL, &sm));
   VkDescriptorSetLayoutBinding lb[3];
   for (int i = 0; i < 3; i++)
      lb[i] = (VkDescriptorSetLayoutBinding){.binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
   VkDescriptorSetLayoutCreateInfo dli = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3, .pBindings = lb};
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dli, NULL, &dsl));
   VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = 4};
   VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl,
                                     .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
   VkPipelineLayout plo;
   CHECK(vkCreatePipelineLayout(dev, &pli, NULL, &plo));
   VkComputePipelineCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = plo,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main"}};
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe));
   VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3};
   VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
   VkDescriptorPool pool;
   CHECK(vkCreateDescriptorPool(dev, &dpi, NULL, &pool));
   VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &dsl};
   VkDescriptorSet set;
   CHECK(vkAllocateDescriptorSets(dev, &dsa, &set));
   VkDescriptorBufferInfo dbi[3] = {{A.b, 0, VK_WHOLE_SIZE}, {B.b, 0, VK_WHOLE_SIZE}, {C.b, 0, VK_WHOLE_SIZE}};
   VkWriteDescriptorSet w[3];
   for (int i = 0; i < 3; i++)
      w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = i, .descriptorCount = 1,
                                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi[i]};
   vkUpdateDescriptorSets(dev, 3, w, 0, NULL);
   VkCommandPoolCreateInfo cpo = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qf};
   VkCommandPool cpool;
   CHECK(vkCreateCommandPool(dev, &cpo, NULL, &cpool));
   VkCommandBufferAllocateInfo cba = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &cba, &cmd));
   VkCommandBufferBeginInfo bgi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(vkBeginCommandBuffer(cmd, &bgi));
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plo, 0, 1, &set, 0, NULL);
   vkCmdPushConstants(cmd, plo, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &rounds);
   vkCmdDispatch(cmd, inv / 64, 1, 1);
   CHECK(vkEndCommandBuffer(cmd));
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(dev, &fci, NULL, &fence));
   double best = 1e9;
   for (int r = 0; r < (bench ? reps : 1); r++) {
      VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
      double t0 = now();
      CHECK(vkQueueSubmit(queue, 1, &si, fence));
      CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 10ull * 1000 * 1000 * 1000));
      double dt = now() - t0;
      if (dt < best) best = dt;
      CHECK(vkResetFences(dev, 1, &fence));
   }
   uint32_t bad = 0;
   for (uint32_t i = 0; i < inv; i++) {
      int32_t want;
      if (!bench) {
         want = sdot(a[i], b[i]) + 7;
      } else {
         int32_t x[8], y[8], s[8] = {0};
         memcpy(x, a + 8 * i, 32); memcpy(y, b + 8 * i, 32);
         for (int r = 0; r < rounds; r++) {
            for (int k = 0; k < 8; k++) s[k] += sdot(x[k], y[k]);
            for (int k = 0; k < 4; k++) { x[k] = (int32_t)((uint32_t)x[k] + 0x01010101u); x[4 + k] = (int32_t)((uint32_t)x[4 + k] - 0x01010101u); }
         }
         want = s[0] ^ s[1] ^ s[2] ^ s[3] ^ s[4] ^ s[5] ^ s[6] ^ s[7];
      }
      if (c[i] != want && bad++ < 3) fprintf(stderr, "  mismatch at %u: GPU %d, CPU %d\n", i, c[i], want);
   }
   printf("%s %s: %u of %u results differ from the CPU", argv[1], argv[2], bad, inv);
   if (bench) {
      double macs = (double)inv * rounds * 8 * 4;
      printf("; best of %d: %.2f ms = %.2f G int8 MAC/s", reps, best * 1e3, macs / best / 1e9);
   }
   printf("\n");
   return bad ? 3 : 0;
}
