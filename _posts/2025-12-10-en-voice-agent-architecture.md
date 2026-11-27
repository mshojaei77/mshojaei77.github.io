---
layout: article
lang: en
title: "Voice Agents in Production: Turn Taking, Interruptions, and Latency"
description: "Design voice agents around the full audio loop: turn detection, cancellation, playback state, tool safety, observability, and task-based evaluation."
date: 2025-12-10 07:35:02 +0000
permalink: /writing/voice-agent-architecture/
nav_exclude: true
article_kind: guide
last_modified_at: 2026-11-27 00:00:00 +0000
author: "Mohammad Shojaei"
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
---

# Voice Agents in Production: Turn Taking, Interruptions, and Latency

A voice agent has to manage more than speech recognition and speech synthesis. It must decide when the user has finished, begin a response at an acceptable speed, stop speaking when interrupted, and keep the conversation state consistent with what the user actually heard.

Those behaviors depend on the complete audio path. A fast language model will not make a slow voice system feel responsive if turn detection waits too long or the text-to-speech service takes several seconds to start. Likewise, a convincing demo can still fail when people pause, backchannel, speak over the agent, or use a noisy connection.

## Two common architectures

Most implementations fall into one of two broad designs.

**A cascaded pipeline** connects separate stages, often voice activity detection (VAD), speech-to-text (STT), an LLM, and text-to-speech (TTS). Each stage can be selected and measured independently. It is easier to replace one component, but the system must pass partial results between stages and coordinate their timing.

**A speech-to-speech model** consumes and produces audio through a real-time conversation interface. It can reduce the number of handoffs and support more direct audio interaction, but the application still owns session state, playback, tool behavior, error recovery, and product-level safety. “End-to-end” does not mean there is no system to engineer.

Choose based on the experience and constraints you need. A phone workflow with a strict transcript audit may favor a cascaded pipeline. A natural spoken interaction with frequent interruptions may benefit from a real-time audio model. Test both against the same user tasks if the decision matters.

## Model the audio turn as state

Do not treat a transcript as a complete representation of a voice interaction. A useful turn lifecycle records at least:

1. Input audio is streaming or being recorded.
2. The system detects speech and decides whether a user turn has ended.
3. A transcript or audio representation is attached to the conversation.
4. The agent generates a response or calls a tool.
5. Output audio is queued and played to the user.
6. Playback completes, is interrupted, or fails.

Every transition needs an owner and an observable event. If the application receives a tool result while the caller has already interrupted the agent, decide whether to cancel the work, keep it running, or return a later result. Cancellation of speech does not undo an external tool action; sensitive or irreversible actions still need authorization checks and, where appropriate, confirmation.

## Turn detection is a product decision

VAD commonly treats a period of silence as the end of a spoken turn. It is simple and can respond quickly, but silence is an imperfect signal. A speaker may pause while thinking, use a long pause between clauses, or speak in a noisy room. Short acknowledgments such as “yeah” may also be mistaken for an interruption.

The [OpenAI Realtime VAD guide](https://developers.openai.com/api/docs/guides/realtime-vad) describes two modes: `server_vad`, which chunks audio around silence, and `semantic_vad`, which uses the words spoken to estimate whether the utterance is complete. A semantic detector can wait longer when a sentence trails off and end a turn sooner after a clear statement. Neither choice is universally correct; measure interruption and response delay with your speakers and environment.

For some interfaces, push-to-talk is the better design. It gives the user control over the recording boundary and can avoid VAD timing errors, at the cost of a less hands-free interaction. The right choice depends on whether people can comfortably press a control during the task.

Tune turn detection together with the interface. A longer endpointing delay can prevent cutting users off but adds silence before each answer. A short delay feels quick but can split a thought into multiple turns. Measure completion quality and time-to-response rather than optimizing a timeout by feel.

## Handle interruptions and playback state

When a person starts speaking over the agent, the system should stop audible output promptly. It then needs to update the conversation so the model does not believe the user heard audio that was still queued on the device.

The exact mechanism depends on the transport. In the [OpenAI Realtime conversation guide](https://developers.openai.com/api/docs/guides/realtime-conversations), a WebRTC or SIP connection lets the server track its output audio buffer and truncate unplayed audio after an interruption. With WebSockets, the client owns playback: it should stop and clear queued audio, measure how much of the response was played, and send a truncation event for the unplayed portion. If application state ignores this distinction, the next response may refer to words that were never heard.

Frameworks expose additional choices. [LiveKit's turn-taking guide](https://docs.livekit.io/agents/logic/turns/) distinguishes VAD-based interruptions from adaptive handling and covers false interruptions, where background sound or a brief acknowledgment stops speech without becoming a real user turn. A system can resume the interrupted response after a false positive, but it must do so without speaking over new user input.

Test interruption behavior explicitly:

- The user says “stop” while the agent is speaking.
- The user begins a correction midway through the response.
- The user gives a short backchannel such as “uh-huh.”
- Background noise triggers speech detection without a transcript.
- The user speaks while a tool call is still running.
- The connection drops after speech was generated but before it finished playing.

## Measure the latency users experience

For a voice interaction, “latency” can mean several different intervals. Start with end-to-end time from the end of the user's speech to the first audible response. Then break the path into the stages that apply to your architecture:

| Stage | Example measurement |
| --- | --- |
| Turn detection | Time from last user speech to turn closure |
| STT | Time to produce the transcript needed for response generation |
| LLM | Time to first output token or tool decision |
| Tool | Time from call to result, including retries |
| TTS | Time to first audio byte |
| Playback | Time from generated audio to audible output on the client |

In a cascaded system, STT, turn detection, LLM, tools, and TTS may all contribute in sequence. A real-time speech model can combine some stages, but its first-audio timing and interruptions still need measurement. [LiveKit session traces](https://docs.livekit.io/reference/agents/session-traces/) describe stage metrics for both pipeline and realtime sessions, including end-to-end latency, STT delay, end-of-turn detection, model time to first token, TTS time to first byte, and tool calls.

Collect latency distributions under realistic concurrency and network conditions. Averages can hide a slow tail that makes users repeat themselves. Also track abandoned turns, false interruptions, repeated user requests, tool failures, and the human effort needed to recover. Faster first audio is not an improvement if it increases cutoffs or produces more repair turns.

## Evaluate the conversation, not only transcripts

Create a task set that includes accents, disfluencies, interruptions, background noise, corrections, silence, and ambiguous requests. Evaluate whether the agent understood the task, handled the turn correctly, used tools safely, and delivered the response the user needed.

The [VoiceBench study](https://aclanthology.org/2026.tacl-1.18/) introduces 6,783 synthetic and real spoken instructions across eight tasks. Its stated focus includes general knowledge, instruction following, and safety under variations such as accents, acoustic conditions, mispronunciations, and content complexity. Such a dataset can inform spoken-input evaluation, but it does not replace measuring your own live audio pipeline, interruption behavior, latency, and tool integration.

For each test case, record the audio conditions, expected task outcome, turn boundaries, any tool action, and grading method. Have humans review a sample of failures because an exact transcript match is often too strict for speech, while a fluent response can still misunderstand the intent. Keep sensitive recordings only when there is a clear retention purpose and an appropriate user notice.

## Guard tool actions and recovery

Voice input can be incomplete or ambiguous, especially with names, amounts, and addresses. Confirm high-impact details before actions that spend money, change account state, or send messages. The language model should not be the only authorization layer for a tool. Validate structured arguments in trusted application code and record whether the action succeeded, failed, or was cancelled.

Make retries idempotent where possible. If a network timeout happens after a payment-like action, blindly repeating the tool call can duplicate the side effect. If the user interrupts while a read-only lookup is running, canceling may be harmless; if a write has already reached a downstream system, the state must be reconciled before another action is attempted.

## Start with the failure you can observe

Instrument the session so you can reconstruct what happened: turn events, response and tool identifiers, stage timings, cancellation reasons, and whether each audio segment was played. Avoid logging more raw audio or sensitive transcript content than needed for debugging.

Then tune one part of the system at a time on the same task set. First establish that turns close at useful points. Next verify interruptions and playback synchronization. Then profile the slow stage and optimize it. This sequence makes it easier to tell whether a latency improvement came from a better model, a shorter endpointing delay, faster tools, or less wasted speech.

For related design work, see [production failure modes for AI agents (in Persian)](/fa/writing/agent-production-failures/) and [agent harness design (in Persian)](/fa/writing/agent-harness/).

## References

- [OpenAI Realtime conversations](https://developers.openai.com/api/docs/guides/realtime-conversations)
- [OpenAI Realtime voice activity detection](https://developers.openai.com/api/docs/guides/realtime-vad)
- [LiveKit turn-taking overview](https://docs.livekit.io/agents/logic/turns/)
- [LiveKit session traces and metrics](https://docs.livekit.io/reference/agents/session-traces/)
- [VoiceBench: Benchmarking LLM-Based Voice Assistants](https://aclanthology.org/2026.tacl-1.18/)
