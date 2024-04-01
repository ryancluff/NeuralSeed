#include "daisy_petal.h"
#include "daisysp.h"
#include "terrarium.h"
#include <RTNeural/RTNeural.h>

// Model Weights (edit this file to add model weights trained with Colab script)
#include "all_model_data.h"

using namespace daisy;
using namespace daisysp;
using namespace terrarium;  // This is important for mapping the correct controls to the Daisy Seed on Terrarium PCB

// Declare a local daisy_petal for hardware access
DaisyPetal hw;
Parameter inLevel, modelParam, modelParam2, modelParam3, outLevel, wetDryMix, noiseFilterFreq, lowPassFilterFreq;
float inLevelf, modelParamf, modelParam2f, modelParam3f, outLevelf, wetDryMixf, lowPassFilterFreqf;

bool                bypass;
int                 modelInSize;
unsigned int        modelIndex;
float brightness1;
float brightness2;

Led led1, led2;

static CrossFade cfade; // For blending the wet/dry signals while maintaining constant volume

// Each EQ will be turned on/off independently
bool       eqOn[2];
int        freqs[2];

// Our bandpass filter for each EQ boost switch
struct Filter
{
    Svf   filt;
    float amp;

    void Init(float samplerate, float freq)
    {
        filt.Init(samplerate);
        filt.SetRes(0.6);
        filt.SetDrive(0.000);
        filt.SetFreq(freq);
        amp = .5f;
    }

    float Process(float in)
    {
        filt.Process(in);
        return filt.Band() * amp;
    }
};

Filter filters[2];
// Filter noiseFilter;
Svf lowPassFilter;

// The center frequency of each filter (change as desired)
//  Currently set at the lower end of the spectrum
void InitFreqs()
{
    freqs[0] = 60;
    freqs[1] = 120;
}

void InitFilters(float samplerate)
{
    for(int i = 0; i < 2; i++)
    {
        filters[i].Init(samplerate, freqs[i]);
    }

    lowPassFilter.Init(samplerate);
}


// These are each of the Neural Model options.
// GRU (Gated Recurrent Unit) networks with input sizes ranging 1 - 4.
// 1 input for audio, up to 3 inputs for parameterized controls, such
// as Gain/Drive or Tone.

RTNeural::ModelT<float, 1, 1,
    RTNeural::GRULayerT<float, 1, 10>,
    RTNeural::DenseT<float, 10, 1>> model;

RTNeural::ModelT<float, 2, 1,
      RTNeural::GRULayerT<float, 2, 10>,
      RTNeural::DenseT<float, 10, 1>> model2;
	  
RTNeural::ModelT<float, 3, 1,
      RTNeural::GRULayerT<float, 3, 8>,
      RTNeural::DenseT<float, 8, 1>> model3;

RTNeural::ModelT<float, 4, 1,
      RTNeural::GRULayerT<float, 4, 8>,
      RTNeural::DenseT<float, 8, 1>> model4;

// Notes: With default settings, GRU 10 is max size currently able to run on Daisy Seed
//        - Parameterized 1-knob GRU 10 is max
//        - Parameterized 2-knob/3-knob at GRU 8 is max

// Loads a new model using the correct template and resets the right LED brightness
void changeModel()
{
    if (bypass) {
       return;
    }
    if (modelIndex == model_collection.size() - 1) {
        modelIndex = 0;
    } else {
        modelIndex += 1;
    }

    if (model_collection[modelIndex].rec_weight_ih_l0.size() == 2) {
      auto& gru = (model2).template get<0>();
      auto& dense = (model2).template get<1>();
      modelInSize = 2;
      gru.setWVals(model_collection[modelIndex].rec_weight_ih_l0);
      gru.setUVals(model_collection[modelIndex].rec_weight_hh_l0);
      gru.setBVals(model_collection[modelIndex].rec_bias);
      dense.setWeights(model_collection[modelIndex].lin_weight);
      dense.setBias(model_collection[modelIndex].lin_bias.data());
      brightness2 = 0.3f;

    } else if (model_collection[modelIndex].rec_weight_ih_l0.size() == 3) {
      auto& gru = (model3).template get<0>();
      auto& dense = (model3).template get<1>();
      modelInSize = 3;
      gru.setWVals(model_collection[modelIndex].rec_weight_ih_l0);
      gru.setUVals(model_collection[modelIndex].rec_weight_hh_l0);
      gru.setBVals(model_collection[modelIndex].rec_bias);
      dense.setWeights(model_collection[modelIndex].lin_weight);
      dense.setBias(model_collection[modelIndex].lin_bias.data());
      brightness2 = 0.65f;

    } else if (model_collection[modelIndex].rec_weight_ih_l0.size() == 4) {
      auto& gru = (model4).template get<0>();
      auto& dense = (model4).template get<1>();
      modelInSize = 4;
      gru.setWVals(model_collection[modelIndex].rec_weight_ih_l0);
      gru.setUVals(model_collection[modelIndex].rec_weight_hh_l0);
      gru.setBVals(model_collection[modelIndex].rec_bias);
      dense.setWeights(model_collection[modelIndex].lin_weight);
      dense.setBias(model_collection[modelIndex].lin_bias.data());
      brightness2 = 1.0f;

    } else {
      auto& gru = (model).template get<0>();
      auto& dense = (model).template get<1>();
      modelInSize = 1;
      gru.setWVals(model_collection[modelIndex].rec_weight_ih_l0);
      gru.setUVals(model_collection[modelIndex].rec_weight_hh_l0);
      gru.setBVals(model_collection[modelIndex].rec_bias);
      dense.setWeights(model_collection[modelIndex].lin_weight);
      dense.setBias(model_collection[modelIndex].lin_bias.data());
      brightness2 = 0.0f;
    }

    // Initialize Neural Net
    if (modelInSize == 1) {
        model.reset();
    } else if (modelInSize == 2) {
        model2.reset();
    } else if (modelInSize == 3) {
        model3.reset();
    } else {
        model4.reset();
    }
}


void UpdateLeds() {
    led1.Set(brightness1);
    led2.Set(brightness2);
    led1.Update();
    led2.Update();
}

void UpdateSwitches()
{

}

void UpdateButtons()
{
    //switch1 pressed
    if(hw.switches[Terrarium::FOOTSWITCH_1].RisingEdge())
    {
        bypass = !bypass;
        brightness1 = bypass ? 0.0f : 1.0f;
    }

    //switch2 pressed
    if(hw.switches[Terrarium::FOOTSWITCH_2].RisingEdge())
    {
        changeModel();
    }
}


void UpdateKnobs()
{
    inLevelf = inLevel.Process();
    outLevelf = outLevel.Process(); 
    wetDryMixf = wetDryMix.Process();
    modelParamf = modelParam.Process();
    modelParam2f = modelParam2.Process();
    modelParam3f = modelParam3.Process();
    lowPassFilterFreqf = lowPassFilterFreq.Process();
}


void Controls()
{
    hw.ProcessAnalogControls();
    hw.ProcessDigitalControls();

    UpdateKnobs();

    UpdateButtons();

    UpdateSwitches();

    UpdateLeds();
}

// This runs at a fixed rate, to prepare audio samples
static void AudioCallback(AudioHandle::InputBuffer  in,
                          AudioHandle::OutputBuffer out,
                          size_t                    size)
{
    float samplerate = hw.AudioSampleRate();

    Controls();

    //hw.ProcessAllControls();
    // hw.ProcessAnalogControls();
    // hw.ProcessDigitalControls();
    // led1.Update();
    // led2.Update();

	float final_mix;

    cfade.SetPos(wetDryMixf);

    float input_arr[4] = { 0.0, 0.0, 0.0, 0.0 };

    // EQ Switches
    //     - The .Pressed() function below counts an 'ON' switch as pressed.
    //     - Each EQ boost is controlled by it's own switch
    int switches[4] = {Terrarium::SWITCH_1, Terrarium::SWITCH_2, Terrarium::SWITCH_3, Terrarium::SWITCH_4}; // Can this be moved elsewhere?
    for(int i=0; i<2; i++)
        eqOn[i] = hw.switches[switches[i]].Pressed();

    for(size_t i = 0; i < size; i++)
    {
        float input = in[0][i];
        float wet   = input;

        // Process your signal here
        if(bypass)
        {
            out[0][i] = in[0][i];
        }
        else
        {
            if (modelInSize == 2) {
                input_arr[0] = input * inLevelf;           // Set input array with input level adjustment
                input_arr[1] = modelParamf;
                wet = model2.forward (input_arr) + input;  // Run Parameterized Model and add Skip Connection
                
            } else if (modelInSize == 3) {
                input_arr[0] = input * inLevelf;
                input_arr[1] = modelParamf;
		        input_arr[2] = modelParam2f;
                wet = model3.forward (input_arr) + input;  // Run Parameterized Model and add Skip Connection

            } else if (modelInSize == 4) {
                input_arr[0] = input * inLevelf;
                input_arr[1] = modelParamf;
		        input_arr[2] = modelParam2f;
                input_arr[3] = modelParam3f;
                wet = model4.forward (input_arr) + input;  // Run Parameterized Model and add Skip Connection

            } else {
                input_arr[0] = input * inLevelf;           
                wet = model.forward (input_arr) + input;   // Run Model and add Skip Connection
            }

            //wet *= 0.6; // Level adjustment, models are too loud

            // wet = wet * wet_dry_mix * 0.2 + input * (1 - wet_dry_mix);  // Set Wet/Dry Mix (and reduce model output)

            // Process EQ 
            float sig = 0.f;
            bool noEQ = true;
            for(int j = 0; j < 2; j++)
            {
                if (eqOn[j]) {
                    sig += filters[j].Process(wet);
                    noEQ = false;
                }
            }
            sig *= .7;  // EQ level adjust if needed

            if(!noEQ) {
                wet += sig;
            }

            if (hw.switches[switches[3]].Pressed()) {
                lowPassFilter.SetFreq(lowPassFilterFreqf * samplerate);
                lowPassFilter.Process(wet);
                wet = lowPassFilter.Notch();
            }

            if (hw.switches[switches[3]].Pressed()) {
                lowPassFilter.SetFreq(lowPassFilterFreqf * samplerate);
                lowPassFilter.Process(wet);
                wet = lowPassFilter.Notch();
            }
            
            
            final_mix = cfade.Process(input, wet); // crossfade wet/dry at constant power (consistent volume)

            out[0][i] = final_mix * outLevelf;                           // Set output level
        }

        // Copy left channel to right channel
	    // Not needed for Terrarium, mono only (left channel)
        //for(size_t i = 0; i < size; i++)
        //{
        //    out[1][i] = out[0][i];
        //}
        
    }
}

int main(void)
{
    float samplerate;
    // float blocksize;

    hw.Init();
    samplerate = hw.AudioSampleRate();
    // blocksize = hw.AudioBlockSize();

    setupWeights();
    // hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_96KHZ);
    // hw.SetAudioBlockSize(48);

    inLevel.Init(hw.knob[Terrarium::KNOB_1], 0.0f, 3.0f, Parameter::LINEAR);
    outLevel.Init(hw.knob[Terrarium::KNOB_3], 0.0f, 1.0f, Parameter::LINEAR); 
    wetDryMix.Init(hw.knob[Terrarium::KNOB_2], 0.0f, 1.0f, Parameter::LINEAR);
    modelParam.Init(hw.knob[Terrarium::KNOB_4], 0.0f, 1.0f, Parameter::LINEAR);
    modelParam2.Init(hw.knob[Terrarium::KNOB_5], 0.0f, 1.0f, Parameter::LINEAR);
    modelParam3.Init(hw.knob[Terrarium::KNOB_6], 0.0f, 1.0f, Parameter::LINEAR);
    lowPassFilterFreq.Init(hw.knob[Terrarium::KNOB_6], 0.0f, 1.0f, Parameter::EXPONENTIAL);

    // modelParam3.Init(hw.knob[Terrarium::KNOB_6], 0.0f, 1.0f, Parameter::LINEAR);

    // Initialize the correct model
    modelIndex = -1;
    changeModel();

    // Initialize filters for 4 EQ boost switches
    InitFreqs();
    InitFilters(samplerate);

    // Initialize & set params for CrossFade object
    cfade.Init();
    cfade.SetCurve(CROSSFADE_CPOW);

    // Init the LEDs and set activate bypass
    led1.Init(hw.seed.GetPin(Terrarium::LED_1),false);
    led1.Update();
    led2.Init(hw.seed.GetPin(Terrarium::LED_2),false);
    led2.Update();

    bypass = true;
    brightness1 = 0.0f;
    brightness2 = 0.1f;


    hw.StartAdc();
    hw.StartAudio(AudioCallback);
    while(1)
    {
        // Do Stuff Infinitely Here
        System::Delay(10);
    }
}