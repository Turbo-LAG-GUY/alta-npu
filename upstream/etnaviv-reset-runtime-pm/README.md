# Sending the NPU reset fix to the etnaviv developers

`0001-drm-etnaviv-keep-the-core-in-reset-across-runtime-po.patch` is a ready-to-send, plain-text kernel patch email
("[RFC PATCH] drm/etnaviv: keep the core in reset across runtime power-down"). It explains the lockup, the evidence
from your board and the fix, and asks the maintainers the open questions. It is marked RFC (request for comments)
because it was tested only on your 6.12.73 kernel, not on mainline.

Nothing has been sent. The kernel's rules for AI-assisted patches (Documentation/process/coding-assistants.rst)
say the AI must never send anything itself, and that only you can sign off.

## 1. Before you send
1. **Read the whole patch.** The email says "I have reviewed the patch", and maintainers may ask you questions
   about it, so only send it once that's true. The kernel's rule is that you must understand everything you submit
   and be able to defend it (generated-content.rst). Plain-English summary of the code change:
   - Release the NPU's reset with longer waits: 10 ms before and 20 ms after, instead of 1 microsecond each.
   - When the NPU goes idle and powers down: hold it in reset after its clocks stop.
   - When it wakes: release the reset, soft-reset it the way startup does, then run the normal setup.
   - If waking fails: stop the clocks and hold the reset again.
   It is the same code as the build you've been running, minus the "keep powered" knob, which upstream doesn't
   need: users can write "on" to power/control themselves.
2. **Add your sign-off.** Put this line directly under the `Fixes:` line, above the first `---`:
   ```
   Signed-off-by: Your Name <your@address>
   ```
   This is your legal statement (the Developer's Certificate of Origin in submitting-patches.rst), which is why
   it's left to you. The kernel wants "a known identity". The patch's `From:` line currently uses your git name,
   Turbo-LAG-GUY. If you'd rather use your real name, change it in both the `From:` line and the sign-off.
3. **Optional:** subscribe to etnaviv@lists.freedesktop.org at https://lists.freedesktop.org/mailman/listinfo/etnaviv.
   The list is moderated for non-members, so without subscribing your post waits until a moderator approves it.

## 2. How to send it (plain text only)
Gmail's web page would reformat the patch and break it. Use `git send-email`:
```
sudo apt install git-email
git config --global sendemail.smtpServer smtp.gmail.com
git config --global sendemail.smtpServerPort 587
git config --global sendemail.smtpEncryption tls
git config --global sendemail.smtpUser <your Gmail address>
```
Gmail needs an app password for this: Google Account, then Security, 2-Step Verification, App passwords.
`git send-email` asks for it when sending. First send it to yourself to check that it arrives intact:
```
cd ~/alta-npu/upstream/etnaviv-reset-runtime-pm
git send-email --to=<your address> 0001-drm-etnaviv-keep-the-core-in-reset-across-runtime-po.patch
```
Then send it for real:
```
git send-email \
  --to="Lucas Stach <l.stach@pengutronix.de>" \
  --cc="Russell King <linux+etnaviv@armlinux.org.uk>" \
  --cc="Christian Gmeiner <christian.gmeiner@gmail.com>" \
  --cc="David Airlie <airlied@gmail.com>" \
  --cc="Simona Vetter <simona@ffwll.ch>" \
  --cc="Philipp Zabel <p.zabel@pengutronix.de>" \
  --cc="Philippe Lecointre <philippe.lecointre@thalesgroup.com>" \
  --cc="Simon Lenain <simon.lenain@thalesgroup.com>" \
  --cc="Tomeu Vizoso <tomeu@tomeuvizoso.net>" \
  --cc="Neil Armstrong <neil.armstrong@linaro.org>" \
  --cc="Kevin Hilman <khilman@baylibre.com>" \
  --cc=etnaviv@lists.freedesktop.org \
  --cc=dri-devel@lists.freedesktop.org \
  --cc=linux-amlogic@lists.infradead.org \
  --cc=linux-kernel@vger.kernel.org \
  0001-drm-etnaviv-keep-the-core-in-reset-across-runtime-po.patch
```
Alternative without SMTP setup: the kernel's `b4` tool can send through kernel.org's web endpoint
(`b4 send --web-auth-new`, see https://b4.docs.kernel.org).

### Who it goes to, and why
- **scripts/get_maintainer.pl**, run with mainline's MAINTAINERS file:
  - Lucas Stach (etnaviv maintainer); Russell King and Christian Gmeiner (etnaviv reviewers).
  - David Airlie and Simona Vetter (DRM maintainers).
  - Philipp Zabel (reset framework).
  - Philippe Lecointre and Simon Lenain (author and acker of the commit named in `Fixes:`).
  - The etnaviv, dri-devel and linux-kernel lists.
- **Added by hand**, because the email asks whether the fix belongs in Amlogic's power-domain driver:
  Neil Armstrong and Kevin Hilman (Amlogic maintainers) and the linux-amlogic list.
- **Also added:** Tomeu Vizoso, who brought up this NPU in etnaviv and Mesa.

## 3. What was checked, and what wasn't
- **Checked:**
  - It applies cleanly to mainline v7.4-rc1 (`git am`). etnaviv_gpu.c is identical in v7.3, v7.4-rc1 and current
    master, so the patch names v7.4-rc1 as its base.
  - `checkpatch.pl --strict`: no style errors or warnings. The only error is the missing sign-off, which you add.
  - The code is the same as the tested build (patches/etnaviv-reset-keepon/0004), apart from the keep-powered knob
    and the wording of one warning message.
- **Not done, and the email says so:**
  - It was not built or run on a mainline kernel.
  - The lockup itself was never captured.
  - The waits are Amlogic's values, untuned.
- **Security classification:** an ordinary bug, not a security hole (Documentation/process/threat-model.rst). It
  happens during normal NPU use, triggering it needs access to the NPU device, and the kernel itself marks the NPU
  interface experimental. So it goes to the public lists, not security@kernel.org.
- **Not included:** your local SH_EU clock-gating patch for 0x7120. We never tested whether it's needed, so it is not
  part of this submission.

## 4. After sending
Replies come by email. Answer below the quoted text you're replying to ("interleaved", never top-posted), and be
patient: 2-3 weeks is normal. Likely questions:
- Can the waits be shorter?
- Should the reset be handled by the power-domain driver instead?
- Can you test on mainline?
I can help you understand the review comments and draft answers. Testing on mainline means building a mainline
kernel for the Alta, which takes hours at -j2.
