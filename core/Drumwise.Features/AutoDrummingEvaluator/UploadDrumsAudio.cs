using Drumwise.Application.Common.Errors;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Models;
using Drumwise.Application.Entities;
using Drumwise.Infrastructure.Data;
using MediatR;
using Microsoft.AspNetCore.Http;
using Polly;
using Polly.Retry;

namespace Drumwise.Features.AutoDrummingEvaluator;

public record UploadDrumsAudioCommand(IFormFile AudioFile) : IRequest<Result>;

public class UploadDrumsAudioHandler(ApplicationDbContext context, IFileStorageService fileStorageService, 
    IConversionService conversionService) 
    : IRequestHandler<UploadDrumsAudioCommand, Result>
{
    public async Task<Result> Handle(UploadDrumsAudioCommand request, CancellationToken cancellationToken)
    {
        if (request.AudioFile.Length == 0)
            return Result.Failure(AutoDrummingEvaluatorErrors.IncorrectDrumsAudio, ResultType.BadRequest);

        var audioFileType = request.AudioFile.ContentType;
        
        var availableTypes = new List<string>()
        {
            "audio/midi", "audio/x-midi", "audio/mid", "audio/mpeg", "audio/wav", "audio/wave", "audio/x-wav"
        };

        if (!availableTypes.Contains(audioFileType))
            return Result.Failure(AutoDrummingEvaluatorErrors.IncorrectDrumsAudio, ResultType.BadRequest);

        if (audioFileType is not ("audio/midi" or "audio/x-midi" or "audio/mid"))
        {
            throw new NotImplementedException();
        }

        var uploadedRecording = new DrumsAudioToEvaluate
        {
            Uploaded = false,
            TimingEvaluationResult = null,
            DynamicsEvaluationResult = null,
            Evaluated = false
        };

        context.DrumsAudiosToEvaluate.Add(uploadedRecording);
        await context.SaveChangesAsync(cancellationToken).ConfigureAwait(false);

        var fileName = uploadedRecording.Id.ToString();

        // TODO: Implement polly 

        await using var recordStream = request.AudioFile.OpenReadStream();
        var (success, fileId) = await fileStorageService.SaveFileAsync(recordStream, fileName, audioFileType, cancellationToken);
        
        if (!success) 
            return Result.Failure(AutoDrummingEvaluatorErrors.UploadingFailed, ResultType.InternalServerError);

        uploadedRecording.Uploaded = true;
        uploadedRecording.GoogleDriveFileId = fileId;
        context.DrumsAudiosToEvaluate.Update(uploadedRecording);
        await context.SaveChangesAsync(cancellationToken).ConfigureAwait(false);

        return Result.Success(ResultType.Ok);
    }
}