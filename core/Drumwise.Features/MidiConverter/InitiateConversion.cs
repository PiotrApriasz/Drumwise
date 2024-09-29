using Drumwise.Application.Common.Errors;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Models;
using Drumwise.Application.Entities;
using Drumwise.Features.AutoDrummingEvaluator;
using Drumwise.Infrastructure.Data;
using MediatR;
using Microsoft.AspNetCore.Http;

namespace Drumwise.Features.MidiConverter;

public record InitiateConversionCommand(IFormFile AudioFile) : IRequest<Result>;

public class InitiateConversionHandler(ApplicationDbContext context, IFileStorageService fileStorageService, 
    IConversionService conversionService) 
    : IRequestHandler<InitiateConversionCommand, Result>
{
    public async Task<Result> Handle(InitiateConversionCommand request, CancellationToken cancellationToken)
    {
        if (request.AudioFile.Length == 0)
            return Result.Failure(MidiConverterErrors.IncorrectDrumsAudio, ResultType.BadRequest);

        var audioFileType = request.AudioFile.ContentType;
        
        var availableTypes = new List<string>()
        {
            "audio/mpeg", "audio/wav", "audio/wave", "audio/x-wav"
        };

        if (!availableTypes.Contains(audioFileType))
            return Result.Failure(MidiConverterErrors.IncorrectDrumsAudio, ResultType.BadRequest);

        var uploadedRecording = new AudioToConvert
        {
            Uploaded = false,
            GDConvertedMidiFileId = null,
            Converted = false
        };

        context.AudiosToConverts.Add(uploadedRecording);
        await context.SaveChangesAsync(cancellationToken);

        var fileName = uploadedRecording.Id.ToString();

        // TODO: Implement polly 

        await using var recordStream = request.AudioFile.OpenReadStream();
        var (success, fileId) = await fileStorageService.SaveFileAsync(recordStream, fileName, audioFileType, cancellationToken);
        
        if (!success) 
            return Result.Failure(MidiConverterErrors.UploadingFailed, ResultType.InternalServerError);

        uploadedRecording.Uploaded = true;
        uploadedRecording.GDAudioFileId = fileId;
        context.AudiosToConverts.Update(uploadedRecording);
        await context.SaveChangesAsync(cancellationToken);

        return Result.Success(ResultType.Ok);
    }
}