namespace Drumwise.Application.Common.Interfaces;

public interface IConversionService
{
    Task<string> ConvertStandardAudioToMidi(string audioFile);
}