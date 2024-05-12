using Drumwise.Application.Common.Errors;

namespace Drumwise.Features.AutoDrummingEvaluator;

public static class AutoDrummingEvaluatorErrors
{
    public static IEnumerable<Error> IncorrectDrumsAudio =>
        Error.ApplicationError(new[] { ErrorCodes.DrumsAudio.IncorrectDrumsAudio });

    public static IEnumerable<Error> UploadingFailed =>
        Error.ApplicationError(new[] { ErrorCodes.DrumsAudio.UploadingFailed });
}