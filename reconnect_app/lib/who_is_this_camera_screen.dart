import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

class WhoIsThisCameraScreen extends StatefulWidget {
  const WhoIsThisCameraScreen({super.key});

  @override
  State<WhoIsThisCameraScreen> createState() => _WhoIsThisCameraScreenState();
}

class _WhoIsThisCameraScreenState extends State<WhoIsThisCameraScreen> {
  CameraController? _controller;
  int _tenthsRemaining = 15;
  String? _error;

  @override
  void initState() {
    super.initState();
    _startCapture();
  }

  Future<void> _startCapture() async {
    try {
      final cameras = await availableCameras();
      final camera = cameras.firstWhere(
        (item) => item.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        camera,
        ResolutionPreset.medium,
        enableAudio: false,
      );
      await controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() => _controller = controller);
      await controller.startVideoRecording();
      for (var value = 14; value >= 0; value--) {
        await Future<void>.delayed(const Duration(milliseconds: 100));
        if (mounted) setState(() => _tenthsRemaining = value);
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
                    'Camera capture failed: $_error',
                    style: const TextStyle(color: Colors.white),
                    textAlign: TextAlign.center,
                  ),
                ),
              )
            : controller == null || !controller.value.isInitialized
            ? const Center(
                child: CircularProgressIndicator(color: Colors.white),
              )
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
                          'Hold the phone steady — capturing ${(_tenthsRemaining / 10).toStringAsFixed(1)}s',
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
