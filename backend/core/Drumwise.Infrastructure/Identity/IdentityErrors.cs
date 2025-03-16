using Drumwise.Application.Common.Errors;
using Microsoft.AspNetCore.Identity;

namespace Drumwise.Infrastructure.Identity;

public static class IdentityErrors
{
    public static IEnumerable<Error> UserNotFound => 
        Error.ApplicationError(new [] {ErrorCodes.Identity.UserNotFound});

    public static IEnumerable<Error> IdentityError(IEnumerable<IdentityError> errors) =>
        Error.IdentityError(errors);

    public static IEnumerable<Error> InvalidEmail(string email) =>
        Error.ApplicationError(new[] { ErrorCodes.Identity.InvalidEmail }, email);
}