import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

/// Dedicated capture for linking a detected unknown voice to a face.
///
/// This is intentionally separate from WhoIsThisCameraScreen: it records a
/// 10-second video with audio for backend active-speaker detection.
class UnknownVoiceCameraScreen extends StatefulWidget {
  const UnknownVoiceCameraScreen({super.key, this.captureSeconds = 10});

  final int captureSeconds;

  @override
  State<UnknownVoiceCameraScreen> createState() => _UnknownVoiceCameraScreenState();
}

class _UnknownVoiceCameraScreenState extends State<UnknownVoiceCameraScreen> {
  CameraController? _controller;
  int _secondsRemaining = 10;
  String? _error;

  @override
  void initState() {
    super.initState();
    _secondsRemaining = widget.captureSeconds;
    _startCapture();
  }

  Future<void> _startCapture() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) throw StateError('No camera is available.');
      final camera = cameras.firstWhere(
        (item) => item.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        camera,
        ResolutionPreset.medium,
        enableAudio: true,
      );
      await controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() => _controller = controller);
      await controller.startVideoRecording();
      for (var value = widget.captureSeconds - 1; value >= 0; value--) {
        await Future<void>.delayed(const Duration(seconds: 1));
        if (mounted) setState(() => _secondsRemaining = value);
      }
      final video = await controller.stopVideoRecording();
      if (mounted) Navigator.of(context).pop(video);
    } catch (error) {
      if (mounted) setState(() => _error = '$error');
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    return PopScope(
      canPop: false,
      child: Scaffold(
        backgroundColor: Colors.black,
        body: _error != null
            ? Center(
                child: Padding(
                  padding: const EdgeInsets.all(24),
                  child: Text(
                    'Unknown voice capture failed: $_error',
                    style: const TextStyle(color: Colors.white),
                    textAlign: TextAlign.center,
                  ),
                ),
              )
            : controller == null || !controller.value.isInitialized
            ? const Center(child: CircularProgressIndicator(color: Colors.white))
            : Stack(
                fit: StackFit.expand,
                children: [
                  CameraPreview(controller),
                  Positioned(
                    top: 56,
                    left: 24,
                    right: 24,
                    child: DecoratedBox(
                      decoration: BoxDecoration(
                        color: Colors.black54,
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Padding(
                        padding: const EdgeInsets.all(12),
                        child: Text(
                          'Capturing voice and video: $_secondsRemaining s remaining',
                          textAlign: TextAlign.center,
                          style: const TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
      ),
    );
  }
}
